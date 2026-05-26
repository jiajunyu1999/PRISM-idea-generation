# Reward Model Release

This directory contains a cleaned, releasable version of the impact reward-model training, evaluation, and serving code for scientific idea judging.

## Project Layout

```text
reward_model_release/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
├── data/
├── models/
├── train/
└── scripts/
```

## Dataset

This project uses **SciJudgeBench**.

Dataset homepage:
- https://modelscope.cn/datasets/openmoss/SciJudgeBench

Place the downloaded dataset files under a directory such as:

```text
path/to/SciJudgeBench/
├── train.jsonl
├── test.jsonl
├── test_ood_year.jsonl
└── test_ood_iclr.jsonl
```

All training and evaluation commands below expect the dataset path to be passed explicitly through `--dataset_dir`.

## Backbone Sources

Backbone weights are fetched from Hugging Face Hub by default. The released configs use the following model identifiers:

```text
allenai/scibert_scivocab_uncased
allenai/scitulu-7b
facebook/galactica-6.7b
Qwen/Qwen2.5-7B-Instruct
meta-llama/Llama-3.1-8B-Instruct
```

If you prefer local checkpoints, replace the `base_model_name` field in the corresponding config with a local path.

## Installation

```bash
pip install -r requirements.txt
```

## Reward Models

### 1. SciBERT

Train:
```bash
CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=8 scripts/train_reward.py \
  --config configs/train_lora_8gpu_stage2_resume_slowdecay.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_full \
  --output_dir outputs/run_scibert
```

Evaluate:
```bash
python scripts/eval_reward.py \
  --config configs/train_lora_8gpu_stage2_resume_slowdecay.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_full \
  --ckpt outputs/run_scibert/best \
  --output_dir outputs/eval_scibert
```

### 2. SciTulu-7B

Train:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 scripts/train_reward_scitulu.py \
  --config configs/train_scitulu7b_lora_8gpu_mainline_tuned.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_scitulu_full_tokenized \
  --output_dir outputs/run_scitulu7b
```

Evaluate:
```bash
python scripts/eval_reward_scitulu.py \
  --config configs/train_scitulu7b_lora_8gpu_mainline_tuned.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_scitulu_full_tokenized \
  --ckpt outputs/run_scitulu7b/best \
  --output_dir outputs/eval_scitulu7b
```

### 3. Galactica-6.7B

Train:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 scripts/train_reward_scitulu.py \
  --config configs/train_galactica67b_lora_8gpu_r8_stable.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_galactica_full_tokenized \
  --output_dir outputs/run_galactica67b
```

Evaluate:
```bash
python scripts/eval_reward_scitulu.py \
  --config configs/train_galactica67b_lora_8gpu_r8_stable.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_galactica_full_tokenized \
  --ckpt outputs/run_galactica67b/best \
  --output_dir outputs/eval_galactica67b
```

### 4. Qwen2.5-7B-Instruct

Train:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 scripts/train_reward_scitulu.py \
  --config configs/train_qwen25_7b_instruct_lora_8gpu_r8.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_qwen25_full_tokenized \
  --output_dir outputs/run_qwen25_7b
```

Evaluate:
```bash
python scripts/eval_reward_scitulu.py \
  --config configs/train_qwen25_7b_instruct_lora_8gpu_r8.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_qwen25_full_tokenized \
  --ckpt outputs/run_qwen25_7b/best \
  --output_dir outputs/eval_qwen25_7b
```

### 5. Meta-Llama-3.1-8B-Instruct

Train:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 scripts/train_reward_scitulu.py \
  --config configs/train_llama31_8b_instruct_lora_8gpu_r8.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_llama31_full_tokenized \
  --output_dir outputs/run_llama31_8b
```

Evaluate:
```bash
python scripts/eval_reward_scitulu.py \
  --config configs/train_llama31_8b_instruct_lora_8gpu_r8.yaml \
  --dataset_dir path/to/SciJudgeBench \
  --processed_dir outputs/processed_llama31_full_tokenized \
  --ckpt outputs/run_llama31_8b/best \
  --output_dir outputs/eval_llama31_8b
```

## Notes

- `scripts/train_reward.py` is the encoder-style training entry for SciBERT.
- `scripts/train_reward_scitulu.py` is the shared decoder-only reward-model training entry for SciTulu, Galactica, Qwen2.5, and Llama-3.1.
- `scripts/eval_reward.py` is used for encoder-style reward models.
- `scripts/eval_reward_scitulu.py` is used for decoder-only reward models.
