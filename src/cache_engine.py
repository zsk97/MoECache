import torch
import collections
from config import CacheConfig
from load_utils import load_switch_expert

from transformers import AutoConfig

class CacheEngine(object):
    def __init__(self, cache_config: CacheConfig, model_config):
        self.cache_size = cache_config.cache_size
        self.num_layers = cache_config.num_layers

        self.model_config = model_config

        self.experts_in_cpu = dict()
        self.experts_in_gpu = dict()

        self.copy_stream = torch.cuda.Stream()

    def init_expert_cpu(self, model_path):
        """ Initialize the expert module in CPU by loading
            and filtering the model state
        """
        model_state = torch.load(model_path, map_location=torch.device('cpu'))

        load_switch_expert(model_state, self.model_config, self.experts_in_cpu, self.num_layers)

        for key, value in self.experts_in_cpu.items():
            print("****************")
            print(key)
            print(value)
        
    def init_expert_gpu(self):
        pass

    def update_pattern(self, new_pattern):
        self.pattern = new_pattern


if __name__ == "__main__":
    model_name = "google/switch-base-16"
    model_config = AutoConfig.from_pretrained(model_name)
    model_state_path = "/home/scratch.shunkangz_gpu/Research/NUS_Project/Checkpoint/models--google--switch-base-16/snapshots/0ef7d88ed50ec5f2cfdc019e81cef04d19700f8f/pytorch_model.bin"
    
    cache_config = CacheConfig(1, model_config.num_layers, "LRU")
    cache_engine = CacheEngine(cache_config, model_config)

    cache_engine.init_expert_cpu(model_state_path)