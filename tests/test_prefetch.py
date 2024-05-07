from MoECache.cache_engine import CacheEngine, CacheConfig
from transformers import AutoConfig

import torch
import threading
import time

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

    start = time.time()
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
    torch.cuda.synchronize()
    end = time.time()

    print("Total time ", end - start)
    cache_engine.exit()
    workerA.join()
    workerB.join()