import torch
import time
from collections import OrderedDict, deque
from .config import CacheConfig
from .load_utils import load_switch_expert, clone_wrapper

import threading
import logging
from multiprocessing import Value
import ctypes
from dataclasses import dataclass
from transformers import AutoConfig
from safetensors.torch import load_file

from enum import Enum

class RequestType(Enum):
    FETCH = 1
    INVALID = 2

@dataclass
class RequestEntry:
    expert_info: tuple[int, int]
    request_type: RequestType

@dataclass
class CallbackEntry:
    expert_info: tuple[int, int]
    cache_pos: int
    finish_event: torch.cuda.Event
    request_type: RequestType

class CacheEngine(object):
    def __init__(self, cache_config: CacheConfig, model_config):
        self.cache_size = cache_config.cache_size
        self.num_layers = cache_config.num_layers
        self.model_config = model_config

        self.experts_in_cpu = OrderedDict()
        self.experts_in_gpu = []

        # Store the map between expert id and the position in GPU cache
        self.expert_to_cache_pos = OrderedDict()

        # Allocate two queues to ensure two types of requests are in FIFO
        self.low_priority_queue = deque()
        self.high_priority_queue = deque()

        self.callback_queue = deque()

        self.expert_in_use = dict()

        self.prev_expert_info = None

        self.cache_lock = threading.Lock()

        self.check_invalid = False
        self.invalid_event = threading.Event()

        self.copy_stream = torch.cuda.Stream()

    def init_expert_cpu(self, model_path):
        """ Initialize the expert module in CPU by loading
            and filtering the model state
        """
        logging.info("Initialize expert in CPU")
        if ".bin" in model_path:
            weight_load_func = lambda filepath, device: torch.load(filepath, map_location=str(device))
        else:
            weight_load_func = lambda filepath, device: load_file(filepath, device=str(device))
        
        model_state = weight_load_func(model_path, torch.device("cpu"))

        load_switch_expert(model_state, self.model_config, self.experts_in_cpu, self.num_layers)
        
        for expert_info in self.experts_in_cpu.keys():
            self.expert_in_use[expert_info] = False

    def init_expert_gpu(self):
        """ Determine the number of experts in the GPU and initialize cache
            Currently, I hard code this part as num_expert_in_gpu and put the
            experts at the beginning of module into GPU cache
        """
        logging.info("Initialize expert in GPU")
        num_expert_in_gpu = 12*32
        device = torch.device("cuda:0")

        if len(self.experts_in_cpu) < num_expert_in_gpu:
            logging.info("All expert can be in GPU cache")
            num_expert_in_gpu = len(self.experts_in_cpu)

        count = 0
        for expert_info, expert_module in self.experts_in_cpu.items():
            self.experts_in_gpu.append(clone_wrapper(expert_module, device))
            self.expert_to_cache_pos[expert_info] = count
            count += 1
            if count == num_expert_in_gpu:
                break

    def update_pattern(self, new_pattern):
        self.pattern = new_pattern

    def prefetch(self, expert_info, high_priority=False):
        """ Put the prefetch request into prefetch queue
            If the high_priority is true, this is on-demand prefetching
            We will insert the request into high_priority queue
        """
        request = RequestEntry(expert_info, RequestType.FETCH)
        if high_priority:
            self.high_priority_queue.append(request)
        else:
            self.low_priority_queue.append(request)

    def _evict(self):
        """ Evict the least recent used expert
            In furture, we can execute the eviction based on predictor
            Check whether the expert is in use
        """
        # TODO: Wait if there is no expert to be evicted
        # If all the cache is in use experts, wait!
        while len(self.expert_to_cache_pos) <= 2:
            continue
        
        # Wait until the expert finish
        # TODO: This check might cause overhead
        evict_expert = list(self.expert_to_cache_pos.keys())[0]
        while True:
            with self.cache_lock:
                if not self.expert_in_use[evict_expert]:
                    break
            logging.debug(f"Expert {evict_expert} in use ", self.expert_in_use[evict_expert])
            time.sleep(0.05)
            evict_expert = list(self.expert_to_cache_pos.keys())[0]

        expert_info, cache_pos = self.expert_to_cache_pos.popitem(last=False)
        logging.debug(f"Evict expert {expert_info}")
        # assert self.expert_in_use[expert_info] == False, "Swapping a in use expert"
        return (expert_info, cache_pos)

    def invalid_experts(self, expert_info):
        """ Invalid the misfetched expert
        """

        # Submit the invalid request into high priority queue
        # If the misfetched expert is in cache,
        # evict it and put it at the head of cache
        callback_entry = CallbackEntry(expert_info, None, None, RequestType.INVALID)
        self.callback_queue.append(callback_entry)

    def exec_request(self):
        """ Fork a new thread to deal with the request
        """
        self.running = True

        while self.running:
            # Check the two queues status
            request = None

            # Block the request queue and wait
            # for the check invalid finish
            # TODO: Insert a spectial request to tell the old and new request
            if self.check_invalid:
                self.invalid_event.wait()

            if len(self.high_priority_queue) != 0:
                request = self.high_priority_queue.popleft()

            elif len(self.low_priority_queue) != 0:
                request = self.low_priority_queue.popleft()

            if request != None:
                match request.request_type:
                    case RequestType.FETCH:
                        # Check if the module already in GPU 
                        if request.expert_info in self.expert_to_cache_pos:
                            logging.debug(f"{request.expert_info} Already in GPU")
                            with self.cache_lock:
                                self.expert_in_use[request.expert_info] = True
                            self._update_lru_cache(request.expert_info)
                        else:
                            logging.debug(f"Evict for {request.expert_info}")
                            _, cache_pos = self._evict()
                            self._copy(request.expert_info, cache_pos)
                        
                            with torch.cuda.stream(self.copy_stream):
                                callback_entry = CallbackEntry(request.expert_info, cache_pos, torch.cuda.Event(), RequestType.FETCH)
                                callback_entry.finish_event.record()
                            self.callback_queue.append(callback_entry)
    
    def exec_callback(self):
        """ Fork a new thread to check the callback queue
        """
        self.running = True

        while self.running:
            callback_entry = None

            if len(self.callback_queue) != 0:
                callback_entry = self.callback_queue.popleft()

                match callback_entry.request_type:
                    case RequestType.FETCH:
                        callback_entry.finish_event.synchronize()
                        self.expert_to_cache_pos[callback_entry.expert_info] = callback_entry.cache_pos
                        with self.cache_lock:
                            logging.debug(f"Expert {callback_entry.expert_info} set True")
                            self.expert_in_use[callback_entry.expert_info] = True
                        self._update_lru_cache(callback_entry.expert_info)

                    case RequestType.INVALID:
                        # Block the request queue
                        self.check_invalid = True

                        # Check the low priority 
                        size_request = len(self.low_priority_queue)
                        count = 0
                        for request in self.low_priority_queue:
                            if request.expert_info == callback_entry.expert_info:
                                logging.debug("Found prefetch request, cancel it!")
                                del self.low_priority_queue[count]
                                break

                            count += 1
                            if count == size_request:
                                break
                        
                        self.check_invalid = False
                        self.invalid_event.set()
                        self.invalid_event.clear()

                        # Check the callback queue
                        if callback_entry.expert_info not in self.expert_to_cache_pos:
                            self.callback_queue.append(callback_entry)
                        else:
                            logging.debug(f"Invalid expert {callback_entry.expert_info}")
                            with self.cache_lock:
                                self.expert_in_use[callback_entry.expert_info] = False
                            self._update_lru_cache(callback_entry.expert_info, False)
                            # logging.debug(f"Expert {callback_entry.expert_info} is in use ", self.expert_in_use[callback_entry.expert_info])

    def exit(self):
        self.running = False
        print("Cache Engine Successfully Exit")
    
    def _mark_unused(self):
        if self.prev_expert_info != None:
            with self.cache_lock:
                self.expert_in_use[self.prev_expert_info] = False

    def load_experts(self, expert_info):
        """ Check if this expert is ready in GPU   
        """
        # TODO: Cannot handle the misfetch expert
        self._mark_unused()
        self.prev_expert_info = expert_info

        # The required expert might be in loading
        while expert_info not in self.expert_to_cache_pos:
            logging.debug("Wait for expert ready")
            continue

        return self.experts_in_gpu[self.expert_to_cache_pos[expert_info]]

    def _copy(self, expert_info, cache_pos):
        torch.cuda.nvtx.range_push("Copy expert")
        with torch.cuda.stream(self.copy_stream):
            self.experts_in_gpu[cache_pos].storage.copy_(self.experts_in_cpu[expert_info].storage, non_blocking=True)
        torch.cuda.nvtx.range_pop()
        
    def _update_lru_cache(self, expert_info, is_tail=True):
        if is_tail:
            self.expert_to_cache_pos.move_to_end(expert_info, last=True)
        else:
            self.expert_to_cache_pos.move_to_end(expert_info, last=False)
    
    def debug_info(self):
        for expert_info in self.expert_to_cache_pos.keys():
            logging.debug("Cache expert ", expert_info)
    
    def get_prefetch_experts(self, layer_id):
        """ This function return the prefetch expert from
            the target layer. Normally, we only require the
            prefetched expert from previous layer
        """
        assert layer_id >= 0 and layer_id < self.pattern.shape[0], "Layer ID exceeds the pattern size"
        expert_list = torch.nonzero(self.pattern[layer_id]).flatten().tolist()
        return expert_list



if __name__ == "__main__":
    model_name = "google/switch-base-16"
    model_config = AutoConfig.from_pretrained(model_name)
    model_state_path = "/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-16/snapshots/0ef7d88ed50ec5f2cfdc019e81cef04d19700f8f/pytorch_model.bin"
    
    cache_config = CacheConfig(1, model_config.num_layers, "LRU")
    cache_engine = CacheEngine(cache_config, model_config)

    cache_engine.init_expert_cpu(model_state_path)
    cache_engine.init_expert_gpu()

    print("Finish setting up cache engine")

    cache_queue = [(1, 5), (1, 9), (1, 7), (3, 8), (3, 10), (3, 4), 
                   (5, 2), (5, 4), (5, 7), (5, 9), (7, 1), (7, 4), 
                   (1, 5), (1, 9), (3, 8)]
    
    compute_stream = torch.cuda.Stream()

    print("Start worker on prefetching queue")
    workerA = threading.Thread(target=cache_engine.exec_request)
    workerA.start()

    workerB = threading.Thread(target=cache_engine.exec_callback)
    workerB.start()

    print("Start loading")
    input = torch.randn((128, model_config.d_model), dtype=torch.bfloat16, device=torch.device("cuda:0"))
    
    cache_engine.prefetch(cache_queue[0])
    num_compute = len(cache_queue)
    count = 1

    with torch.cuda.stream(compute_stream):
        for expert_info in cache_queue:
            print("Calculating expert ", expert_info)
            module = cache_engine.load_experts(expert_info)

            if count < num_compute:
                cache_engine.prefetch(cache_queue[count])
                count += 1
            for i in range(5):
                res = module(input)

    cache_engine.exit()
    workerA.join()
    workerB.join()