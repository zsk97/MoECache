import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init
import threading

import os
import logging
from accessory.util import misc
from transformers import AutoTokenizer
from glob import glob
from datasets import load_dataset

from MoECache.build_model import build_switch_offload_model
from MoECache.load_utils import process_dataset
from MoECache.generate import fix_decode_generate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    model_name = "google/switch-base-32"
    model_path = "/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-32/snapshots/2018338b8dad760fa7a35a754d532486ef3942f9/pytorch_model.bin"
    model, cache_engine = build_switch_offload_model(model_name, model_path)
    model = model.bfloat16().to(device).eval()


    # Start the cache_engine
    workerA = threading.Thread(target=cache_engine.exec_request)
    workerA.start()

    workerB = threading.Thread(target=cache_engine.exec_callback)
    workerB.start()

    # Load data
    dataset = load_dataset("marsggbo/bigbench4switch32_pattern_predictor")
    batch_size = 8

    logging.info("Start Inference")

    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-16")
    tokenizer.padding_side = 'left'
    batch_id = 0
    for input_data, decode_id, pattern in process_dataset(dataset, tokenizer, batch_size):
        logging.info(f"Inference for Batch {batch_id}")
        input_ids = input_data.input_ids.to(device)
        attention_mask = input_data.attention_mask.to(device)
        decode_input_id = decode_id.to(device)
        predict_pattern = pattern.to(device)

        output = fix_decode_generate(input_ids, decode_input_id, attention_mask, predict_pattern, model, cache_engine)


    cache_engine.exit()
    workerA.join()
    workerB.join()