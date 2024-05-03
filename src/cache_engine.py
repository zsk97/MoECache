import torch
from collections import OrderedDict, deque
from config import CacheConfig
from load_utils import load_switch_expert, clone_wrapper

from dataclasses import dataclass
from transformers import AutoConfig

# @dataclass
# class RequestEntry:



class CacheEngine(object):
    def __init__(self, cache_config: CacheConfig, model_config):
        self.cache_size = cache_config.cache_size
        self.num_layers = cache_config.num_layers
        self.model_config = model_config

        self.experts_in_cpu = OrderedDict()
        self.experts_in_gpu = []

        # Store the map between expert id and the position in GPU cache
        self.expert_to_cache_pos = OrderedDict()

        self.prefetch_queue = deque()

        self.copy_stream = torch.cuda.Stream()

    def init_expert_cpu(self, model_path):
        """ Initialize the expert module in CPU by loading
            and filtering the model state
        """
        model_state = torch.load(model_path, map_location=torch.device('cpu'))

        load_switch_expert(model_state, self.model_config, self.experts_in_cpu, self.num_layers)
        
    def init_expert_gpu(self):
        """ Determine the number of experts in the GPU and initialize cache
            Currently, I hard code this part as num_expert_in_gpu and put the
            experts at the beginning of module into GPU cache
        """
        num_expert_in_gpu = 5
        device = torch.device("cuda:0")

        if len(self.experts_in_cpu) < num_expert_in_gpu:
            print("All expert can be in GPU cache")
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

    def prefetch(self, layer_idx, expert_idx, high_priority=False):
        """ 
        """

    def evict(self):
        """
        """


if __name__ == "__main__":
    model_name = "google/switch-base-16"
    model_config = AutoConfig.from_pretrained(model_name)
    model_state_path = "/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-16/snapshots/0ef7d88ed50ec5f2cfdc019e81cef04d19700f8f/pytorch_model.bin"
    
    cache_config = CacheConfig(1, model_config.num_layers, "LRU")
    cache_engine = CacheEngine(cache_config, model_config)

    cache_engine.init_expert_cpu(model_state_path)
    cache_engine.init_expert_gpu()