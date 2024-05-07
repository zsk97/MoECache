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

    real_queue = [[(1, 2), (1, 3), (1, 5)],
                  [(3, 7), (3, 9), (3, 11)],
                  [(5, 1), (5, 3), (5, 6)]]

    prefetch_queue = [[(1, 2), (1, 4), (1, 5)],
                      [(3, 3), (3, 9), (3, 11)],
                      [(5, 2), (5, 3), (5, 5)]]
    
    compute_stream = torch.cuda.Stream()

    print("Start worker on prefetching queue")
    workerA = threading.Thread(target=cache_engine.exec_request)
    workerA.start()

    workerB = threading.Thread(target=cache_engine.exec_callback)
    workerB.start()

    print("Start loading")
    input = torch.randn((128, model_config.d_model), dtype=torch.bfloat16, device=torch.device("cuda:0"))
    
    num_compute = len(real_queue)
    count = 1

    start = time.time()
    for expert_info in prefetch_queue[0]:
        cache_engine.prefetch(expert_info)

    with torch.cuda.stream(compute_stream):
        for i in range(num_compute):
            # Compare the real pattern and prefetch pattern
            # Launch the on-demand prefetch immediately
            set_real = set(real_queue[i])
            set_prefetch = set(prefetch_queue[i])

            mis_fetch = set_prefetch - set_real
            ondemand_fetch = set_real - set_prefetch
            correct_fetch = set_prefetch & set_real

            # Invalid the misfetch experts
            for expert_info in mis_fetch:
                cache_engine.invalid_experts(expert_info)

            # On demand load the experts
            for expert_info in ondemand_fetch:
                cache_engine.prefetch(expert_info, high_priority=True)

            # Add the prefetch of next layer into queue
            if i + 1 < num_compute:
                for expert_info in prefetch_queue[i+1]:
                    cache_engine.prefetch(expert_info)

            for expert_info in correct_fetch:
                module = cache_engine.load_experts(expert_info)
                for j in range(2):
                    res = module(input)
            
            for expert_info in ondemand_fetch:
                module = cache_engine.load_experts(expert_info)
                for j in range(2):
                    res = module(input)

    torch.cuda.synchronize()
    end = time.time()

    print("Total time ", end - start)
    cache_engine.exit()
    workerA.join()
    workerB.join()