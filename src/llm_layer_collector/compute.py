import copy
import torch
from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import DynamicCache

from llm_layer_collector.auto.auto_rotary import AutoRotaryEmbedding
from llm_layer_collector.state_obj import LLmComputationState

def compute_embedding(
        input_embedder: torch.nn.Embedding,
        input_ids: torch.Tensor,
        config: PretrainedConfig,
        state: Optional[LLmComputationState] = None
    ) -> LLmComputationState:
    device = input_embedder.weight.device
    embedded_input = input_embedder(input_ids.to(device))
    if state is None:
        state = LLmComputationState()
    
    state.state = embedded_input
    
    if config.use_cache and state.past_key_values is None:
        state.past_key_values = DynamicCache()
    
    past_seen_tokens = state.past_key_values.get_seq_length() if state.past_key_values is not None else 0
    state.cache_position = torch.arange(
        past_seen_tokens, end=past_seen_tokens + embedded_input.size(1), device=device
    )
    
    state.position_ids = state.cache_position.unsqueeze(0)

    mask_kwargs = {
        "config": config,
        "input_embeds": embedded_input.detach(),
        "attention_mask": None,
        "cache_position": state.cache_position,
        "past_key_values": state.past_key_values,
        "position_ids": state.position_ids
    }
    
    state.causal_mask["full_attention"] = create_causal_mask(**mask_kwargs)

    try:
        if "sliding_attention" in config.layer_types or config.sliding_window is not None:
            state.causal_mask["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    except AttributeError:
        pass

    if config.model_type == 'gemma3_text':
        state.position_embeddings_global = AutoRotaryEmbedding(config)(embedded_input.detach(), state.position_ids)
        configCopy = copy.deepcopy(config)
        configCopy.rope_theta = configCopy.rope_local_base_freq
        configCopy.rope_scaling = {"rope_type": "default"}
        
        state.position_embeddings_local = AutoRotaryEmbedding(configCopy)(embedded_input.detach(), state.position_ids)
    else:
        state.position_embeddings = AutoRotaryEmbedding(config)(embedded_input.detach(), state.position_ids)
    
    return state

def compute_head(
        head: torch.nn.Linear,
        state: torch.Tensor,
        topk: int = 1
    ) -> torch.Tensor:
    state = head(state[:, -1, :])
    probs = torch.softmax(state, dim=-1)
    return torch.topk(probs, topk).indices
