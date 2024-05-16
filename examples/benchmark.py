import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init
import threading

import os
from accessory.util import misc
from transformers import AutoTokenizer
from glob import glob

from MoECache.build_model import build_switch_offload_model

def init_env():
    # define the model
    misc.init_distributed_mode()
    fs_init.initialize_model_parallel(dist.get_world_size())

if __name__ == '__main__':
    init_env()
    rank = dist.get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    if os.environ.get('ipdb', False):
        from ipdb import set_trace
        set_trace()

    model_name = "google/switch-base-16"
    model_path = "/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-16/snapshots/0ef7d88ed50ec5f2cfdc019e81cef04d19700f8f/pytorch_model.bin"
    model, cache_engine = build_switch_offload_model(model_name, model_path)
    model = model.bfloat16().to(device).eval()

    # Start the cache_engine
    workerA = threading.Thread(target=cache_engine.exec_request)
    workerA.start()

    workerB = threading.Thread(target=cache_engine.exec_callback)
    workerB.start()

    x = torch.randint(0, 100, (8, 40)).to(device)
    attention_mask = torch.ones(8, 40).to(device)
    decoder_input_ids=torch.tensor([[0]]*len(x)).int().to(device)

    model(input_ids=x, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)

    cache_engine.exit()
    workerA.join()
    workerB.join()