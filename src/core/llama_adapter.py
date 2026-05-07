"""
LLaMA 3.2 1B backbone adapter for BiCAP-GPT.
Replaces GPT-2 with a more capable language model.

Key differences from GPT-2:
- hidden_size: 2048 (vs 768)
- layers: 16 (vs 12)
- Uses Grouped-Query Attention (GQA) + RoPE

Requires projection layers to match the rest of the model (768 dim).

Updates (2026-03):
- Added MLP projection option to mitigate linear bottleneck
- MLP projection: 768 -> 1024 -> GELU -> 2048 (more expressive mapping)
- Added direct_llm_io mode: when the perceiver outputs directly to 2048, skip projections
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import torch.nn as nn

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from typing import Optional, List
from transformers import AutoModel, AutoTokenizer, AutoConfig


class MLPProjection(nn.Module):
    """
    Non-linear MLP projection to better leverage LLaMA's larger hidden dimension.

    Mitigates the linear bottleneck where a single linear layer
    cannot fully transform 768-dim features to use 2048-dim capacity.

    Architecture:
        Input (768) -> Linear (1024) -> GELU -> Dropout -> Linear (2048) -> LayerNorm
    """
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class LLaMA32WithPrompts(nn.Module):
    """
    LLaMA 3.2 1B model with text prompt support.

    Includes projection layers to maintain compatibility with
    the rest of the pipeline which uses 768-dim embeddings.

    Architecture (linear projection):
        Input (768) -> Linear (2048) -> LLaMA -> Linear (768)

    Architecture (MLP projection - recommended):
        Input (768) -> MLP (768->1024->2048) -> LLaMA -> MLP (2048->1024->768)
    """

    def __init__(self, causal=0, lora=False, ln_grad=True, layers=None,
                 use_text_prompts=False, input_dim=768, use_mlp_proj=False,
                 proj_dropout=0.1, direct_llm_io=False):
        super().__init__()

        # LLaMA 3.2 1B hidden size (set first, needed below)
        self.llama_hidden_size = 2048

        # direct_llm_io: When the perceiver outputs directly to 2048, skip projections
        # This removes the linear/MLP projection bottleneck entirely
        self.direct_llm_io = direct_llm_io

        # The pipeline uses 768-dim embeddings for tokenizers
        # But when direct_llm_io is True, LLM input/output is 2048
        self.input_dim = self.llama_hidden_size if direct_llm_io else input_dim
        self.emb_dim = input_dim  # For compatibility with tokenizers (always 768)

        self.use_text_prompts = use_text_prompts
        self.use_lora = lora
        self.use_mlp_proj = use_mlp_proj and not direct_llm_io  # MLP proj not needed if direct

        # Model ID
        model_id = "meta-llama/Llama-3.2-1B"

        # HuggingFace token for gated models (set via HF_TOKEN env var or `huggingface-cli login`)
        hf_token = os.environ.get("HF_TOKEN", None)

        print(f"Loading LLaMA 3.2 1B model...")
        print(f"  Model: {model_id}")

        # Load config first to check dimensions
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        self.llama_hidden_size = config.hidden_size
        print(f"  Hidden size: {self.llama_hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")

        # Load model
        self.llm = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for training stability
            trust_remote_code=True,
            token=hf_token
        )

        # Truncate layers if specified
        if layers is not None and layers < config.num_hidden_layers:
            self.llm.layers = self.llm.layers[:layers]
            print(f"  Truncated to {layers} layers")

        # Freeze all parameters first
        for param in self.llm.parameters():
            param.requires_grad = False

        # Unfreeze specific components based on strategy
        unfrozen_count = 0
        if ln_grad:
            for name, param in self.llm.named_parameters():
                # LLaMA uses different naming: input_layernorm, post_attention_layernorm
                if 'layernorm' in name.lower() or 'ln' in name.lower() or 'norm' in name.lower():
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"  Unfroze {unfrozen_count} LayerNorm parameters")

        # LoRA support (optional, requires peft library)
        if lora:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                )
                self.llm = get_peft_model(self.llm, lora_config)
                print(f"  LoRA enabled (r=8, alpha=16)")
            except ImportError:
                print(f"  Warning: peft not installed, LoRA disabled")
                self.use_lora = False

        # Projection layers: 768 <-> 2048
        if direct_llm_io:
            # Direct LLM I/O: perceiver outputs 2048, LLM outputs 2048
            # No projection needed - use identity layers
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
            self.proj_in_ln = nn.Identity()
            self.proj_out_ln = nn.Identity()
            print(f"  Direct LLM I/O: perceiver outputs 2048 -> LLaMA -> 2048 (no projection)")
        elif use_mlp_proj:
            # Non-linear MLP projection (recommended for LLaMA)
            proj_hidden = (input_dim + self.llama_hidden_size) // 2
            self.proj_in = MLPProjection(
                input_dim, self.llama_hidden_size,
                hidden_dim=proj_hidden, dropout=proj_dropout
            )
            self.proj_out = MLPProjection(
                self.llama_hidden_size, input_dim,
                hidden_dim=proj_hidden, dropout=proj_dropout
            )
            # MLP already includes LayerNorm
            self.proj_in_ln = nn.Identity()
            self.proj_out_ln = nn.Identity()
            print(f"  MLP Projection: {input_dim} -> {proj_hidden} -> {self.llama_hidden_size} -> {proj_hidden} -> {input_dim}")
        else:
            # Simple linear projection (original)
            self.proj_in = nn.Linear(input_dim, self.llama_hidden_size)
            self.proj_out = nn.Linear(self.llama_hidden_size, input_dim)
            # Layer norm for projections
            self.proj_in_ln = nn.LayerNorm(self.llama_hidden_size)
            self.proj_out_ln = nn.LayerNorm(input_dim)
            print(f"  Linear Projection: {input_dim} -> {self.llama_hidden_size} -> {input_dim}")

        # Tokenizer for text prompts
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_text_prompts:
            print("  Text prompts enabled")
        else:
            print("  Text prompts disabled")

    def forward(self, x: torch.FloatTensor, attention_mask=None,
                text_prompts: Optional[List[str]]=None):
        """
        Forward pass with optional text prompts.

        Args:
            x: (B, seq_len, 768) - ST token embeddings
            attention_mask: Optional attention mask
            text_prompts: Optional[List[str]] - Text prompts (length B)

        Returns:
            out: (B, seq_len, 768) - Hidden states (same dim as input)
        """
        B, seq_len, _ = x.shape
        device = x.device

        # Project input: 768 -> 2048
        x_proj = self.proj_in(x)
        x_proj = self.proj_in_ln(x_proj)

        if self.use_text_prompts and text_prompts is not None:
            # Tokenize text prompts
            encoded = self.tokenizer(
                text_prompts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            text_attn_mask = encoded['attention_mask'].to(device)

            # Get text embeddings from LLaMA
            text_embeds = self.llm.embed_tokens(input_ids)  # (B, M, 2048)
            M = text_embeds.size(1)

            # Concatenate: [text | ST]
            combined_embeds = torch.cat([text_embeds, x_proj], dim=1)

            # Combined attention mask
            if attention_mask is not None:
                combined_mask = torch.cat([text_attn_mask, attention_mask], dim=1)
            else:
                st_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
                combined_mask = torch.cat([text_attn_mask, st_mask], dim=1)

            # Pass through LLaMA
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                output_hidden_states=False
            )
            full_output = outputs.last_hidden_state  # (B, M+seq_len, 2048)

            # Extract only ST outputs
            llama_out = full_output[:, M:, :]  # (B, seq_len, 2048)
        else:
            # No text prompts
            outputs = self.llm(
                inputs_embeds=x_proj,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            llama_out = outputs.last_hidden_state  # (B, seq_len, 2048)

        # Project output: 2048 -> 768
        out = self.proj_out(llama_out)
        out = self.proj_out_ln(out)

        return out

    def get_embedding(self, x: torch.LongTensor):
        """Get token embeddings from the vocabulary."""
        embeds = self.llm.embed_tokens(x)
        # Project to 768 dim
        return self.proj_out(embeds)

    def get_tokenizer(self):
        """Return the tokenizer instance."""
        return self.tokenizer


# Test
if __name__ == "__main__":
    print("Testing LLaMA32WithPrompts...")

    # Check if model is accessible
    try:
        # Test 1: Without text prompts
        print("\n[Test 1] Loading model without text prompts...")
        model = LLaMA32WithPrompts(layers=4, use_text_prompts=False)

        B, N, D = 2, 10, 768
        x = torch.randn(B, N, D)

        print(f"  Input shape: {x.shape}")
        output = model(x)
        print(f"  Output shape: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        print("  Passed!")

        # Test 2: With text prompts
        print("\n[Test 2] Testing with text prompts...")
        model_prompts = LLaMA32WithPrompts(layers=4, use_text_prompts=True)

        prompts = ["Monday 08:00 morning rush hour traffic", "Friday 18:00 evening rush"]
        output_prompts = model_prompts(x, text_prompts=prompts)

        assert output_prompts.shape == x.shape
        print(f"  Prompts: {prompts}")
        print(f"  Output shape: {output_prompts.shape}")
        print("  Passed!")

        # Print parameter count
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[Stats]")
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Trainable ratio: {trainable/total*100:.2f}%")

        print("\nAll tests passed!")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: LLaMA 3.2 requires authentication with Hugging Face.")
        print("Run: huggingface-cli login")
        print("And accept the license at: https://huggingface.co/meta-llama/Llama-3.2-1B")
