import torch

def fix_decode_generate(
        input_ids,
        decode_ids,
        attention_mask,
        predict_pattern,
        model,
        cache_engine,
        max_new_tokens=128,
        past_key_values=None,
        temperature=0.9,
        top_p=0.9
    ):
    device = input_ids.device

    # 初始化生成的令牌列表和past_key_values（用于存储注意力层的状态，加速和优化生成）
    generated_tokens = []
    past = past_key_values
    model.eval()  # Put model in evaluation mode

    # TODO: The first token does not have pattern
    pattern = torch.ones((24, 32), dtype=torch.int).to(device)
    decoder_input_ids = torch.tensor([[0]]*len(input_ids)).int().to(device)
    encoder_outputs = None

    with torch.no_grad():  # Disable gradient calculation
        for step in range(max_new_tokens):
            # Set the cache_engine prefetch pattern and launch
            # prefetch request for the first encoder layer
            if encoder_outputs is None:
                cache_engine.update_pattern(pattern)
                next_layer_experts = cache_engine.get_prefetch_experts(1)
                for expert_id in next_layer_experts:
                    cache_engine.prefetch((1, expert_id))
            else:
            # We prefetch the first decode layer
                cache_engine.update_pattern(pattern)
                next_layer_experts = cache_engine.get_prefetch_experts(12+1)

            outputs = model(input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past,
                            encoder_outputs=encoder_outputs,
                            output_router_logits=True,
                            use_cache=True)  # use_cache允许模型返回past_key_values

            # Select the next token based on the decode_id
            next_token = decode_ids[:, step]

            # 将生成的令牌添加到列表和解码器输入中
            generated_tokens.append(next_token)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            # Update the predict pattern
            pattern = predict_pattern[step]

            # Update Key-Value cache
            past = model_outputs.past_key_values


        return torch.cat(generated_tokens, dim=-1), (model_outputs.encoder_router_logits, model_outputs.decoder_router_logits)