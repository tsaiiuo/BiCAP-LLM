# BiCAP-LLM

**Bidirectional Cross-Attention Perceiver with Large Language Models for Spatial-Temporal Traffic Forecasting**

BiCAP-LLM leverages pre-trained LLMs (GPT-2 / LLaMA) for traffic flow prediction by compressing spatial sensor graphs through bidirectional cross-attention and injecting semantic context via data-driven text prompts.

## Architecture

BiCAP-LLM consists of five stages:

1. **Spatial-Temporal Embedding** — Historical traffic input (B, T, N, F) is augmented with temporal embeddings (day-of-week + time-of-day via learned lookup tables) and spectral node embeddings (Laplacian eigenvectors projected through a linear layer) to encode both temporal periodicity and spatial graph topology.

2. **BiCAP Spatial Perceiver** — A bidirectional cross-attention perceiver (BiXT, NeurIPS'24) compresses N sensor nodes into M latent tokens via cross-attention encoding, then reconstructs back to N nodes via cross-attention decoding with a gating mechanism. This reduces the spatial dimension for efficient LLM processing while preserving graph structure.

3. **Statistical Prompt Generator** — Automatically constructs natural language prompts from the input data at four levels: *simple* (time-of-day + traffic period label), *enhanced* (detailed traffic descriptions with expectations), *task* (task-oriented instructions with node count and prediction horizon), and *task_enhanced* (task instructions enriched with real-time batch statistics such as mean flow, standard deviation, peak hour ratios, and congestion indicators). These prompts are tokenized and embedded alongside traffic tokens to provide semantic guidance to the LLM.

4. **LLM Backbone with Partial Frozen Attention** — Compressed node tokens, temporal tokens (state + gradient), and prompt embeddings are fed into a pre-trained LLM (GPT-2 or LLaMA). Self-attention weights are frozen while LayerNorm and LoRA adapters remain trainable, preserving the LLM's pre-trained knowledge while adapting to the traffic domain.

5. **Prediction Head** — A two-layer MLP decodes the LLM hidden states into future traffic flow predictions (B, P, N, F).

## Requirements

- Python 3.8+
- PyTorch 2.4+
- CUDA 11.8+
- transformers (HuggingFace)

```bash
conda activate ST-PEMLLM
```

## Quick Start

```bash
cd src

# PEMS04 — Full training with BiCAP-LLM baseline
python train.py \
    --data_path ../data/traffic/PEMS04/PEMS04.npz \
    --adj_filename ../data/traffic/PEMS04/PEMS04.csv \
    --dataset PEMS04FLOW \
    --model gpt2 \
    --spatial_attn \
    --use_bidirectional \
    --use_text_prompts \
    --prompt_level task_enhanced \
    --lora --ln_grad \
    --llm_layers 3 \
    --node_embedding --time_token \
    --latent_dim 128 --num_latents 128 \
    --t_dim 64 --node_emb_dim 64 \
    --batch_size 64 --lr 0.0005 \
    --epoch 500 --patience 50 \
    --seed 42
```

## Project Structure

```
BiCAP-LLM/
├── src/
│   ├── train.py                 # Training entry point
│   ├── model/
│   │   └── bicap.py             # Main model: BiCAPForecaster
│   ├── core/
│   │   ├── bicap_module.py      # BiCAP perceiver (BiXT encode/decode)
│   │   ├── gpt2_adapter.py      # GPT-2 with LoRA + PFA + text prompts
│   │   ├── llama_adapter.py     # LLaMA 3.2 1B adapter
│   │   ├── vanilla_transformer.py  # Ablation baseline (no pre-training)
│   │   ├── prompt_builder.py    # Text prompt generator
│   │   └── sinusoidal_pe.py     # Positional encoding
│   ├── pipeline/
│   │   ├── data_factory.py      # Data loading pipeline
│   │   ├── loader.py            # Dataset & sliding window
│   │   └── preprocessing.py     # Data preprocessing
│   └── helpers/
│       ├── config.py            # CLI argument parsing
│       ├── evaluation.py        # MAE, RMSE, MAPE metrics
│       ├── graph_ops.py         # Adjacency, Laplacian, shortest paths
│       ├── misc.py              # Masking utilities
│       └── visualization.py     # Training curve plots
└── data/
    └── traffic/
        ├── PEMS03/
        ├── PEMS04/
        ├── PEMS07/
        └── PEMS08/
```

## Data Preparation

The PEMS traffic datasets are not included in this repository. Download and place them as follows:

1. Download PEMS03, PEMS04, PEMS07, PEMS08 from a public data repository.

2. Organize under `data/traffic/`:
```
data/traffic/
├── PEMS03/
│   ├── PEMS03.npz    # Flow data (T, N, F)
│   └── PEMS03.csv    # Adjacency matrix
├── PEMS04/
│   ├── PEMS04.npz
│   └── PEMS04.csv
├── PEMS07/
│   ├── PEMS07.npz
│   └── PEMS07.csv
└── PEMS08/
    ├── PEMS08.npz
    └── PEMS08.csv
```

Each `.npz` file contains a `data` key with shape `(T, N, F)` where T is the number of time steps, N is the number of sensor nodes, and F is the number of features. Each `.csv` file contains the weighted adjacency matrix.

| Dataset | Nodes | Time Steps | Interval |
|---------|-------|-----------|----------|
| PEMS03  | 358   | 26,208    | 5 min    |
| PEMS04  | 307   | 16,992    | 5 min    |
| PEMS07  | 883   | 28,224    | 5 min    |
| PEMS08  | 170   | 17,856    | 5 min    |

## Supported LLM Backbones

| Backbone | Flag | Notes |
|----------|------|-------|
| GPT-2 (124M) | `--model gpt2` | Default, 12-layer / 3-layer subset |
| LLaMA 3.2 (1B) | `--model llama` | Larger backbone, MLP projection |
| Vanilla Transformer | `--model transformer` | Ablation (no pre-training) |

## Citation

```
TBD
```
