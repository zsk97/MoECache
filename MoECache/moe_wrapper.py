import torch
import torch.nn as nn

from fairscale.nn.model_parallel.layers import (
    reduce_from_model_parallel_region
)

class SwitchMoEWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, cache_engine):
        config.num_experts_per_tok = config.num_selected_experts
        config.intermediate_size = config.d_ff
        config.num_local_experts = config.num_experts
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.num_layer = config.num_layers
        self.layer_id = layer_id
        self.router = gate
        self.cache_engine = cache_engine
        self.token_pattern_mask = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)
        active_experts = expert_index.flatten().unique().tolist()
        active_experts = set(active_experts)

        # Obtain the previous prefetched experts
        prefetch_experts = self.cache_engine.get_prefetch_experts(self.layer_id)
        prefetch_experts = set(prefetch_experts)

        misfetch_experts = prefetch_experts - active_experts
        ondemand_experts = active_experts - prefetch_experts
        correct_experts = active_experts & prefetch_experts

        # Abort the misfetched request
        for expert_id in misfetch_experts:
            self.cache_engine.invalid_experts((self.layer_id, expert_id))

        # Launch the on-demand fetch request
        for expert_id in ondemand_experts:
            self.cache_engine.prefetch((self.layer_id, expert_id), high_priority=True)

        # Launch the prefetch for next layer, except for the last layer
        # TODO: Optimize the last layer prefetch 
        if self.layer_id != self.num_layer - 1:
            next_layer_experts = self.cache_engine.get_prefetch_experts(self.layer_id+1)
            for expert_id in next_layer_experts:
                self.cache_engine.prefetch((self.layer_id+1, expert_id))
        
        # First calculate the correct_experts and then the on-demand experts
        expert_list = list(correct_experts) + list(ondemand_experts)

        next_states = torch.zeros_like(hidden_states)
        for expert_id in expert_list:
            expert_module = self.cache_engine.load_experts((self.layer_id, expert_id))
            token_indices = router_mask[:, :, expert_id].bool()
            if torch.any(token_indices):
                expert_out = expert_module(hidden_states[token_indices]).to(next_states.dtype)
                next_states[token_indices] = expert_out * router_probs[token_indices]
        hidden_states = reduce_from_model_parallel_region(next_states)
        return hidden_states, (router_logits, expert_index)
