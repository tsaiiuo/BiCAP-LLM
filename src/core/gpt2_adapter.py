"""
GPT-2 backbone with text prompt support (HuggingFace version)
Uses the standard transformers library for model loading.

Learnable prompt strategies based on:
- Prompt Tuning (Lester et al., 2021): Simple learnable soft prompts
- Prefix-Tuning (Li & Liang, 2021): MLP reparameterization for stable training
- P-Tuning (Liu et al., 2021): Trainable continuous prompts
"""

import torch
import torch.nn as nn
from typing import Optional, List
from transformers import GPT2Model, GPT2Tokenizer


class PromptEncoder(nn.Module):
    """
    MLP-based prompt encoder for stable training (Prefix-Tuning style).

    Rather than directly learning high-dimensional embeddings, learn a smaller
    representation and project it through an MLP.
    """
    def __init__(self, prompt_len: int, hidden_dim: int, bottleneck_dim: int = 128):
        super().__init__()
        self.prompt_len = prompt_len
        self.hidden_dim = hidden_dim

        # Learnable input (smaller dimension for stable training)
        self.prompt_tokens = nn.Parameter(torch.randn(prompt_len, bottleneck_dim) * 0.02)

        # MLP to project to full hidden dimension
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.Tanh(),
            nn.Linear(bottleneck_dim * 2, hidden_dim)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """Produce prompt embeddings for the given batch size."""
        # (prompt_len, bottleneck) -> (prompt_len, hidden_dim)
        prompt_embeds = self.mlp(self.prompt_tokens)
        # Expand to batch: (1, prompt_len, hidden_dim) -> (B, prompt_len, hidden_dim)
        return prompt_embeds.unsqueeze(0).expand(batch_size, -1, -1)


class GPT2WithPrompts(nn.Module):
    """
    GPT-2 model with text prompt support (HuggingFace version).

    Supports multiple learnable prompt modes for ablation study:
    - 'random': Simple random initialization (baseline)
    - 'vocab': Initialize from random vocabulary embeddings
    - 'text_init': Initialize from template text, then fine-tune
    - 'mlp': MLP reparameterization (Prefix-Tuning style, most stable)
    """
    def __init__(self, causal=0, lora=False, ln_grad=True, layers=None, use_text_prompts=False,
                 use_learnable_prompt=False, learnable_prompt_len=16, prompt_init_mode='random'):
        super().__init__()

        self.emb_dim = 768
        self.use_text_prompts = use_text_prompts
        self.use_learnable_prompt = use_learnable_prompt
        self.learnable_prompt_len = learnable_prompt_len
        self.prompt_init_mode = prompt_init_mode

        # Load pretrained GPT-2
        print(f"Loading GPT-2 model (layers={layers if layers else 'all'})...")
        self.llm = GPT2Model.from_pretrained('gpt2')

        # Truncate to specified number of layers
        if layers is not None:
            self.llm.h = self.llm.h[:layers]
            print(f"  Truncated to {layers} layers")

        # Freeze all parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # Unfreeze LayerNorm and position embeddings (ln_grad strategy)
        if ln_grad:
            for name, param in self.llm.named_parameters():
                if 'ln_' in name or 'ln_f' in name or 'wpe' in name:
                    param.requires_grad = True
            print(f"  Unfroze LayerNorm + position embeddings")

        # Standard tokenizer (for compatibility)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Text prompt support
        if self.use_text_prompts:
            self.text_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            print("  Text prompts enabled")
        else:
            self.text_tokenizer = None

        # Learnable prompt for ablation study
        self.learnable_prompt = None
        self.prompt_encoder = None

        if self.use_learnable_prompt:
            self._init_learnable_prompt()

        if not self.use_text_prompts and not self.use_learnable_prompt:
            print("  No prompts (baseline without prompt)")

    def _init_learnable_prompt(self):
        """Initialize learnable prompt based on the selected mode."""

        if self.prompt_init_mode == 'random':
            # Simple random initialization (original approach)
            self.learnable_prompt = nn.Parameter(
                torch.randn(1, self.learnable_prompt_len, self.emb_dim) * 0.02
            )
            print(f"  Learnable prompt [random]: {self.learnable_prompt_len} tokens")

        elif self.prompt_init_mode == 'vocab':
            # Initialize from random vocabulary embeddings
            # Leverages pre-trained knowledge from GPT-2's word embeddings
            vocab_size = self.llm.wte.weight.size(0)
            random_ids = torch.randint(0, vocab_size, (self.learnable_prompt_len,))
            init_embeds = self.llm.wte.weight[random_ids].detach().clone()
            self.learnable_prompt = nn.Parameter(init_embeds.unsqueeze(0))
            print(f"  Learnable prompt [vocab]: {self.learnable_prompt_len} tokens (init from vocab)")

        elif self.prompt_init_mode == 'text_init':
            # Initialize from a template text, then fine-tune
            # Uses semantic meaning as a starting point
            template = "Traffic flow prediction task for spatial temporal data analysis"
            encoded = self.tokenizer(
                template,
                return_tensors='pt',
                padding='max_length',
                max_length=self.learnable_prompt_len,
                truncation=True
            )
            init_embeds = self.llm.wte(encoded['input_ids']).detach().clone()
            self.learnable_prompt = nn.Parameter(init_embeds)
            print(f"  Learnable prompt [text_init]: '{template[:30]}...'")

        elif self.prompt_init_mode == 'mlp':
            # MLP reparameterization (Prefix-Tuning style)
            # More stable training, especially for longer prompt sequences
            self.prompt_encoder = PromptEncoder(
                prompt_len=self.learnable_prompt_len,
                hidden_dim=self.emb_dim,
                bottleneck_dim=128
            )
            print(f"  Learnable prompt [mlp]: {self.learnable_prompt_len} tokens (Prefix-Tuning style)")

        else:
            raise ValueError(f"Unknown prompt_init_mode: {self.prompt_init_mode}")

    def _get_learnable_prompt_embeds(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Retrieve learnable prompt embeddings based on the active mode."""
        if self.prompt_init_mode == 'mlp':
            # Use MLP encoder
            prompt_embeds = self.prompt_encoder(batch_size).to(device)
        else:
            # Direct learnable parameter
            prompt_embeds = self.learnable_prompt.expand(batch_size, -1, -1)
        return prompt_embeds

    def forward(self, x: torch.FloatTensor, attention_mask=None, text_prompts: Optional[List[str]]=None):
        """
        Forward pass with optional text prompts or learnable prompts.

        Args:
            x: (B, seq_len, emb_dim) - ST token embeddings
            attention_mask: Optional attention mask
            text_prompts: Optional[List[str]] - Text prompts (length B)

        Returns:
            out: (B, seq_len, emb_dim) - Hidden states
        """
        B = x.size(0)
        device = x.device

        # Case 1: Use text prompts
        if self.use_text_prompts and text_prompts is not None:
            # Tokenize text prompts
            encoded = self.text_tokenizer(
                text_prompts,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            text_attn_mask = encoded['attention_mask'].to(device)

            # Get text embeddings
            prompt_embeds = self.llm.wte(input_ids)  # (B, M, emb_dim)
            M = prompt_embeds.size(1)

            # Add positional embeddings
            position_ids = torch.arange(0, M, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
            prompt_pos_embeds = self.llm.wpe(position_ids)
            prompt_embeds = prompt_embeds + prompt_pos_embeds

            prompt_attn_mask = text_attn_mask

        # Case 2: Use learnable prompts (for ablation study)
        elif self.use_learnable_prompt and (self.learnable_prompt is not None or self.prompt_encoder is not None):
            M = self.learnable_prompt_len

            # Get learnable prompt embeddings based on the active mode
            prompt_embeds = self._get_learnable_prompt_embeds(B, device)  # (B, M, emb_dim)

            # Add positional embeddings for learnable prompt tokens
            position_ids = torch.arange(0, M, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
            prompt_pos_embeds = self.llm.wpe(position_ids)
            prompt_embeds = prompt_embeds + prompt_pos_embeds

            # All learnable prompt tokens are valid (mask = 1)
            prompt_attn_mask = torch.ones(B, M, dtype=torch.long, device=device)

        # Case 3: No prompts
        else:
            outputs = self.llm(inputs_embeds=x, attention_mask=attention_mask)
            out = outputs.last_hidden_state
            return out

        # Concatenate: [prompt | ST]
        combined_embeds = torch.cat([prompt_embeds, x], dim=1)  # (B, M+seq_len, emb_dim)

        # Combined attention mask
        if attention_mask is not None:
            combined_mask = torch.cat([prompt_attn_mask, attention_mask], dim=1)
        else:
            st_mask = torch.ones(B, x.size(1), dtype=torch.long, device=device)
            combined_mask = torch.cat([prompt_attn_mask, st_mask], dim=1)

        # Pass through GPT-2
        outputs = self.llm(inputs_embeds=combined_embeds, attention_mask=combined_mask)
        full_output = outputs.last_hidden_state  # (B, M+seq_len, emb_dim)

        # Extract only ST outputs
        out = full_output[:, M:, :]  # (B, seq_len, emb_dim)

        return out

    def get_embedding(self, x: torch.LongTensor):
        """Get token embeddings from the vocabulary."""
        return self.llm.wte(x)

    def get_tokenizer(self):
        """Return the tokenizer instance."""
        return self.tokenizer


# Test
if __name__ == "__main__":
    print("Testing GPT2WithPrompts (HuggingFace version)...")

    # Test 1: Without text prompts
    print("\n[Test 1] Without text prompts")
    model = GPT2WithPrompts(layers=3, use_text_prompts=False)

    B, N, D = 2, 10, 768
    x = torch.randn(B, N, D)

    output = model(x)
    assert output.shape == x.shape
    print(f"  Input: {x.shape}, Output: {output.shape}")
    print("  Passed!")

    # Test 2: With text prompts
    print("\n[Test 2] With text prompts")
    model_prompts = GPT2WithPrompts(layers=3, use_text_prompts=True)

    prompts = ["Mon 08:00 morning rush", "Fri 18:00 evening rush"]
    output_prompts = model_prompts(x, text_prompts=prompts)

    assert output_prompts.shape == x.shape
    print(f"  Prompts: {prompts}")
    print(f"  Input: {x.shape}, Output: {output_prompts.shape}")
    print("  Passed!")

    print("\nAll tests passed!")
