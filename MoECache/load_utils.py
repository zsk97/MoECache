import torch
import torch.nn as nn

import copy 

from .expert_wrapper import SwitchExpertWrapper
from .switch_transformer import SwitchTransformersDenseActDense

# ****************************************************
# ************** Load Utils for Expert ************** 
# ****************************************************

def extract_switch_expert_name(weight_name, num_layer):
    """ Hard code for the switch transformer checkpoint name """
    names = weight_name.split(".")
    expert_id = names[-3].split("_")[-1]
    layer_id = names[2]

    # Avoid same layer index for encoder and decoder
    if names[0] == 'decoder':
        layer_id += num_layer
    return (layer_id, expert_id)

def load_switch_expert(model_state, model_config, experts_in_cpu, num_layer):
    device = torch.device("cpu")

    for layer_idx in range(1, num_layer, 2):
        # Deal with encoder expert
        for expert_idx in range(0, model_config.num_experts):
            expert_name = f"encoder.block.{layer_idx}.layer.1.mlp.experts.expert_{expert_idx}"
            expert = SwitchTransformersDenseActDense(model_config).bfloat16()

            for weight in ["wi", "wo"]:
                layer = getattr(expert, weight)
                ckpt_name = f"{expert_name}.{weight}.weight"
                assert ckpt_name in model_state.keys(), f"Find mismathced expert name {ckpt_name}"
                ckpt = model_state[ckpt_name]
                layer.weight.data.copy_(ckpt)

            experts_in_cpu[(layer_idx, expert_idx)] = SwitchExpertWrapper(expert, device)
            experts_in_cpu[(layer_idx, expert_idx)].storage = experts_in_cpu[(layer_idx, expert_idx)].storage.pin_memory()
        
        # Deal with decoder expert
        for expert_idx in range(0, model_config.num_experts):
            expert_name = f"decoder.block.{layer_idx}.layer.2.mlp.experts.expert_{expert_idx}"
            expert = SwitchTransformersDenseActDense(model_config).bfloat16()

            for weight in ["wi", "wo"]:
                layer = getattr(expert, weight)
                ckpt_name = f"{expert_name}.{weight}.weight"
                assert ckpt_name in model_state.keys(), f"Find mismathced expert name {ckpt_name}"
                ckpt = model_state[ckpt_name]
                layer.weight.data.copy_(ckpt)

            # Avoid conflict layer index
            storage_layer_idx = layer_idx + num_layer
            experts_in_cpu[(storage_layer_idx, expert_idx)] = SwitchExpertWrapper(expert, device)
            experts_in_cpu[(storage_layer_idx, expert_idx)].storage = experts_in_cpu[(storage_layer_idx, expert_idx)].storage.pin_memory()

def clone_wrapper(wrapper: SwitchExpertWrapper, device):
    expert = copy.deepcopy(wrapper.expert_module)
    expert_gpu = SwitchExpertWrapper(expert, device)

    return expert_gpu

# ****************************************************
# ************** Load Utils for Dataset ************** 
# ****************************************************

def process_dataset(dataset, tokenizer, batch_size):
    len_dataset = len(dataset['train'])
    num_batch = len_dataset // batch_size
    num_moe_layer = 6
    num_expert = 32

    for i in range(num_batch):
        prompts = []
        decode_id = []
        decode_pattern = []

        # Extract the batch info
        for j in range(batch_size):
            sample = dataset['train'][i*batch_size+j]
            prompts.append(sample['prompt_text'])
            decode_id.append(sample['decode_ids'])
            decode_pattern.append(sample['decode_pattern'])
        
        # Padding prompts
        input_data = tokenizer(prompts, return_tensors="pt", padding=True, return_attention_mask=True)

        decode_id = torch.Tensor(decode_id)
        decode_length = decode_id.shape[-1]

        # Deal with pattner
        decode_pattern = torch.Tensor(decode_pattern)
        decode_pattern = decode_pattern.permute((2, 1, 0))
        
        pattern = torch.zeros((decode_length, num_moe_layer, num_expert), dtype=torch.int)
        for token_id in range(decode_length):
            for j in range(num_moe_layer):
                batch_pattern = decode_pattern[token_id][j].to(int).flatten().unique().tolist()
                pattern[token_id][j][batch_pattern] = 1

        
        yield input_data, decode_id, pattern