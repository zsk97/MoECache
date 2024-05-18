import torch
from transformers import AutoConfig
import logging
from safetensors.torch import load_file

from MoECache.config import CacheConfig
from MoECache.cache_engine import CacheEngine
from MoECache.utils import with_default_dtype
from MoECache.switch_transformer import (SwitchTransformersForConditionalGeneration, 
                                        SwitchTransformersAttention, SwitchTransformersLayerFF,
                                        SwitchTransformersSparseMLP)
from MoECache.moe_wrapper import SwitchMoEWrapper

def forward_pre_hook(module, input):
    if isinstance(module, SwitchMoEWrapper):
        torch.cuda.nvtx.range_push(f"Layer ID {module.layer_id} {module.__class__.__name__}")
    else:
        torch.cuda.nvtx.range_push(f"Layer ID {module.__class__.__name__}")

def forward_post_hook(module, input, output):
    torch.cuda.nvtx.range_pop()

def build_switch_offload_model(model_name, model_path):
    device = torch.device("cuda:0")
    model_config = AutoConfig.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    cache_config = CacheConfig(1, model_config.num_layers, "LRU")
    cache_engine = CacheEngine(cache_config, model_config)

    # TODO: Only support SwtichTransformer
    cache_engine.init_expert_cpu(model_path)
    cache_engine.init_expert_gpu()

    logging.info("Finish setting up cache engine")

    with device, with_default_dtype(torch.bfloat16):
        model = SwitchTransformersForConditionalGeneration(model_config)
    
    for block_type in ["encoder", "decoder"]:
        if block_type == 'encoder':
            num_block_layers = model_config.num_layers
            sparse_step = model_config.encoder_sparse_step
            block_inner_layer_id = 1
            base_layer_idx = 0
        else:
            num_block_layers = model_config.num_decoder_layers
            sparse_step = model_config.decoder_sparse_step
            block_inner_layer_id = 2
            base_layer_idx = model_config.num_layers
        for block_idx in list(range(num_block_layers))[1:][::sparse_step]:
            curr_layer = getattr(model, block_type).block[block_idx].layer[block_inner_layer_id]
            curr_layer.mlp = SwitchMoEWrapper(
                config=model_config,
                layer_id=block_idx+base_layer_idx,
                gate=curr_layer.mlp.router,
                cache_engine=cache_engine,
            )
    
    logging.info("Finish replacing MoE layer")
    
    # Load model state except experts
    # TODO: Avoid load model static twice
    if ".bin" in model_path:
        weight_load_func = lambda filepath, device: torch.load(filepath, map_location=str(device))
    else:
        weight_load_func = lambda filepath, device: load_file(filepath, device=str(device))
    
    model_state_dict = weight_load_func(model_path, device)
    non_expert_dict = {}
    for key, val in model_state_dict.items():
        if "expert" not in key:
            non_expert_dict[key] = val
    model.load_state_dict(non_expert_dict, True)

    # Add hook for model
    for module in model.modules():
        if isinstance(module, SwitchMoEWrapper) or isinstance(module, SwitchTransformersAttention) or \
        isinstance(module, SwitchTransformersLayerFF) or isinstance(module, SwitchTransformersSparseMLP):
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_post_hook)
    
    return model, cache_engine
