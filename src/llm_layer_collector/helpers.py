import os

import torch
from safetensors import safe_open
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask

def load_shard_tensor(
        layer_file_cache: dict, 
        model_dir: str,
        layer_name: str, 
        device: str,
        dtype: torch.dtype
    ) -> torch.Tensor:
    if layer_name not in layer_file_cache:
        raise ValueError(f'Could not find layer file for layer {layer_name}')
    file = layer_file_cache[layer_name]
    shard: dict = safe_open(os.path.join(model_dir, file), framework='pt', device=device)
    return shard.get_tensor(layer_name).to(dtype)

# Changed in https://github.com/huggingface/transformers/pull/37866
def update_causal_mask(
        config: PretrainedConfig,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor
    ) -> torch.Tensor:
    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    return create_causal_mask(
        config=config,
        input_embeds=input_tensor,
        cache_position=cache_position,
        past_key_values=None,
        attention_mask=None
    )