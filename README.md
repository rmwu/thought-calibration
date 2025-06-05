# Thought calibration

Official implementation of thought calibration (under review).

## Overview

Reasoning large language models achieve impressive test-time scaling by thinking
for longer, but this performance gain comes at significant compute cost.
Directly limiting test-time budget hurts overall performance, but not all
problems are equally difficult. We propose thought calibration to decide
dynamically when thinking can be terminated. To calibrate our decision rule,
we view a language model's growing body of thoughts as a nested sequence of
reasoning trees, where the goal is to identify the point at which novel
reasoning plateaus. We realize this framework through lightweight probes
that operate on top of the language model's hidden representations, which
are informative of both the reasoning structure and overall consistency of
response.

If you find our work interesting, please check out our paper to learn more:
[Thought calibration: Efficient and confident test-time scaling
](https://arxiv.org/abs/2505.18404).

## Installation

```
conda create -y --name tc pip python=3.12
conda activate tc

# basic packages
pip install tqdm numpy pandas matplotlib seaborn scikit-learn jupyter

# model weights and datasets
pip install transformers datasets

# NOTE: customize to your hardware/CUDA configuration

## by default, we run our experiments using lmdeploy
pip install lmdeploy==0.7.3

## if you prefer to use vllm, you can install this instead of lmdeploy
# pip install vllm --extra-index-url https://download.pytorch.org/whl/cu126
```

Thought calibration was tested using Python 3.12 with CUDA 12.6 on A6000 GPUs.

## Code distribution

Our workflows are documented under `notebooks`.

- `1-prepare_s1` contains prompts for budget forcing and model output
  validation.
- `2-probe` contains code for training prompts.
- `3-calibrate` contains code for calibrating probes.
- `4-apply` contains code for applying probes.

Our scripts for running LLMs (budget forcing, inference, verifier) can be found under `src`.

## Data distribution

You may download our calibrated probe weights and thresholds
[here](https://figshare.com/articles/dataset/s1K_calibrated_probes/29242328).

Data required to reproduce these probes are available
[here](https://figshare.com/articles/dataset/s1K_step_embeddings/29230682)

