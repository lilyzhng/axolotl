# Qwen3-VL Fine-tuning with Axolotl

This example shows how to fine-tune [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) using LoRA on the [MaskGroups-HQ dataset](https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ).

## Two Training Modes

1. **Mask Group Prediction** (`lora-8b.yaml`) - Train to predict mask groups from queries
2. **Verifier / Quality Gate** (`verifier-lora-8b.yaml`) - Train ACCEPT/REJECT classifier for segmentation QA

---

## Option 1: Mask Group Prediction

## Quick Start

### 1. Setup Environment

```bash
# Install axolotl
pip install -e .

# Install additional dependencies
pip install matplotlib wandb
```

### 2. Download and Prepare Dataset

The MaskGroups-HQ dataset stores images separately. Download and preprocess:

```bash
# Create data directory
mkdir -p ./data/maskgroups-hq
cd ./data/maskgroups-hq

# Download images (8.06 GB)
wget https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ/resolve/main/images_resized.zip

# Extract (or use Python if unzip not available)
unzip images_resized.zip
# Alternative: python -c "import zipfile; zipfile.ZipFile('images_resized.zip', 'r').extractall('.')"

cd ../..

# Preprocess dataset
python scripts/custom/preprocess_maskgroups.py \
    --images-dir ./data/maskgroups-hq/images_resized \
    --output-dir ./data/maskgroups-hq
```

### 3. Login to Weights & Biases (Optional but Recommended)

```bash
wandb login
```

### 4. Run Training

```bash
PYTHONPATH=. accelerate launch -m axolotl.cli.train examples/qwen3-vl/lora-8b.yaml
```

## Config Highlights

```yaml
# Model
base_model: Qwen/Qwen3-VL-8B-Instruct
processor_type: AutoProcessor
chat_template: qwen2_vl

# LoRA
adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05

# Training
micro_batch_size: 4
gradient_accumulation_steps: 16  # Effective batch size: 64
learning_rate: 0.0001
num_epochs: 1

# W&B + Image Logging
wandb_project: qwen3-vl-finetuning
plugins:
  - scripts.custom.wandb_image_callback.WandbImagePlugin
```

## Expected Results

Training on a single H100 GPU:
- **Training time**: ~20 minutes for 1 epoch
- **Training loss**: 4.66 → 0.99
- **Eval loss**: 6.26 → 1.03

## Visualize Training

### Option 1: W&B Dashboard
If W&B is enabled, view at: https://wandb.ai/YOUR_USERNAME/qwen3-vl-finetuning

### Option 2: Local Plot
```bash
python scripts/custom/plot_training_loss.py \
    --log-file ./outputs/qwen3-vl-verifier/debug.log
```

## Files

- `lora-8b.yaml` - Main training config
- `../../scripts/custom/preprocess_maskgroups.py` - Dataset preprocessing
- `../../scripts/custom/wandb_image_callback.py` - W&B image logging plugin
- `../../scripts/custom/plot_training_loss.py` - Local loss plotting

---

## Option 2: Verifier Training (ACCEPT/REJECT)

Train a quality gate model that verifies if a mask group matches a query.

### 1. Create Verifier Dataset

This creates overlay images with masks rendered on them, labeled ACCEPT or REJECT:

```bash
# Install dependencies
pip install pycocotools scipy

# Create verifier dataset (takes ~15 min for full dataset)
python scripts/custom/create_verifier_dataset.py \
    --images-dir ./data/maskgroups-hq/images_resized \
    --output-dir ./data/verifier-dataset \
    --num-negatives 4

# This creates:
# - 1 ACCEPT per sample (correct mask group)
# - 4 REJECT per sample (2 hard negatives from same image, 2 easy from other images)
# - Total: ~18,000 samples (3599 × 5)
```

### 2. Train Verifier

```bash
PYTHONPATH=. accelerate launch -m axolotl.cli.train examples/qwen3-vl/verifier-lora-8b.yaml
```

### Verifier Dataset Format

Each sample is an overlay image (original + mask highlighted) with a simple prompt:

```
System: You are a segmentation QA verifier.
User: Query: {query}. Does the highlighted mask group match the query? Answer exactly: ACCEPT or REJECT.
Assistant: ACCEPT  (or REJECT)
```

### Verifier Config Highlights

```yaml
# Short responses (just ACCEPT/REJECT)
sequence_len: 256

# Classification-style training
num_epochs: 1  # v0: start with 1 epoch

# Track classification metrics
eval_causal_lm_metrics:
  - accuracy
```

### Deployment Usage

```python
# Given candidate masks from SAM/segmenter:
# 1. Render overlay image
# 2. Ask verifier: ACCEPT/REJECT
# 3. ACCEPT → auto-pass, REJECT → human review
```

---

## Files

| File | Description |
|------|-------------|
| `lora-8b.yaml` | Mask group prediction training |
| `verifier-lora-8b.yaml` | ACCEPT/REJECT verifier training |
| `../../scripts/custom/preprocess_maskgroups.py` | Dataset preprocessing |
| `../../scripts/custom/create_verifier_dataset.py` | Create verifier dataset with overlays |
| `../../scripts/custom/wandb_image_callback.py` | W&B image logging plugin |
| `../../scripts/custom/plot_training_loss.py` | Local loss plotting |

## References

- [Qwen3-VL Model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [MaskGroups-HQ Dataset](https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ)
- [Axolotl Multimodal Docs](https://docs.axolotl.ai/docs/multimodal.html)

