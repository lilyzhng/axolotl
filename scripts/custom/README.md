# Custom Scripts for Axolotl VLM Fine-tuning

This folder contains custom scripts for fine-tuning Vision Language Models (VLMs) with axolotl.

## Scripts

### 1. `preprocess_maskgroups.py`
Preprocesses the [MaskGroups-HQ dataset](https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ) for use with axolotl.

**What it does:**
- Converts image filenames to full local paths
- Converts ShareGPT/Vicuna format (`conversations`) to OpenAI format (`messages`)
- Wraps image paths in a list (required by axolotl)
- Saves as JSON for training

**Usage:**
```bash
# First, download and extract images
mkdir -p ./data/maskgroups-hq
cd ./data/maskgroups-hq
wget https://huggingface.co/datasets/Shengcao1006/MaskGroups-HQ/resolve/main/images_resized.zip
unzip images_resized.zip

# Then run preprocessing
cd ../..
python scripts/custom/preprocess_maskgroups.py \
    --images-dir ./data/maskgroups-hq/images_resized \
    --output-dir ./data/maskgroups-hq
```

### 2. `wandb_image_callback.py`
Custom axolotl plugin for logging images and predictions to Weights & Biases during VLM training.

**Features:**
- Logs sample input images at training start
- Generates and logs model predictions vs ground truth at each evaluation step
- Creates comparison tables in W&B

**Usage:**
Add to your axolotl config:
```yaml
# Enable W&B
wandb_project: your-project-name
wandb_watch: gradients
wandb_name: your-run-name

# Add the plugin
plugins:
  - scripts.custom.wandb_image_callback.WandbImagePlugin
```

Then run with PYTHONPATH set:
```bash
PYTHONPATH=. accelerate launch -m axolotl.cli.train your-config.yaml
```

### 3. `plot_training_loss.py`
Simple script to plot training and eval loss from axolotl's debug.log file.

**Usage:**
```bash
python scripts/custom/plot_training_loss.py \
    --log-file ./outputs/your-run/debug.log \
    --output ./outputs/your-run/training_loss.png
```

## Example Config

See `examples/qwen3-vl/lora-8b.yaml` for a complete config using these scripts to fine-tune Qwen3-VL-8B on MaskGroups-HQ.

## Requirements

```bash
pip install matplotlib wandb datasets tqdm
```

