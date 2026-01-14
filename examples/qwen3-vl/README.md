# Qwen3-VL Fine-tuning with Axolotl

This example shows how to fine-tune [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) using LoRA on the [MaskGroups-HQ dataset](https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ).

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

## References

- [Qwen3-VL Model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [MaskGroups-HQ Dataset](https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ)
- [Axolotl Multimodal Docs](https://docs.axolotl.ai/docs/multimodal.html)

