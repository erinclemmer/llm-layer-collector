import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm

mapper = {
    "llama": LlamaRMSNorm,
    "qwen3": Qwen3RMSNorm,
    "gemma3_text": Gemma3RMSNorm
}

def getClass(config: PretrainedConfig) -> torch.nn.Module:
    return mapper[config.model_type]

class AutoRMSNorm:
    def __init__(self, config: PretrainedConfig):
        self.config = config
        class_type = getClass(config)
        self.cls = class_type(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.cls(hidden_states).to(hidden_states.dtype)

    def to(self, device: str) -> 'AutoRMSNorm':
        self.cls = self.cls.to(device)
        return self
