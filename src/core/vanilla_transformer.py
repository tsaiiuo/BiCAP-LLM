"""
Vanilla Transformer backbone - minimal LLM module for compatibility.
Use gpt2_adapter.GPT2WithPrompts for GPT-2 based experiments.
"""

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """Vanilla Transformer encoder (lightweight baseline)."""
    def __init__(self, causal=0, lora=False, ln_grad=True, layers=None):
        super().__init__()

        self.emb_dim = 768

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=12)
        self.llm = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3)

    def forward(self, x: torch.FloatTensor, attention_mask=None, text_prompts=None):
        # text_prompts is ignored - vanilla Transformer does not support text prompts
        out = self.llm(x)
        return out


# Placeholder classes for unavailable model backends
class Phi2:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phi2 is not available. Use gpt2 instead.")


class LLAMA3:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LLAMA3 is not available. Use gpt2 instead.")
