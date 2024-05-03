import torch
from expert_wrapper import SwitchExpertWrapper
from switch_transformer import SwitchTransformersDenseActDense

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


