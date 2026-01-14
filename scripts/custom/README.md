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

### 4. `create_verifier_dataset.py`
Creates a verifier training dataset with ACCEPT/REJECT labels from MaskGroups-HQ.

**What it does:**
- Creates overlay images with masks highlighted
- Generates positive (ACCEPT) samples with correct mask groups
- Generates hard negatives (REJECT) using other semantic objects from the same image
- Generates easy negatives (REJECT) using masks from different images

**Usage:**
```bash
python scripts/custom/create_verifier_dataset.py \
    --images-dir ./data/maskgroups-hq/images_resized \
    --output-dir ./data/verifier-dataset \
    --num-negatives 4 \
    --hard-negative-ratio 0.5
```

### 5. `visualize_verifier_dataset.py`
Creates comparison figures to visualize ACCEPT vs REJECT samples.

**Usage:**
```bash
python scripts/custom/visualize_verifier_dataset.py \
    --dataset-dir ./data/verifier-dataset \
    --output ./data/verifier-dataset/comparison_figure.png \
    --num-samples 3
```

### 6. `analyze_coverage.py`
Analyzes mask overlay coverage statistics for the verifier dataset.

**Usage:**
```bash
# Analyze all samples with detailed output
python scripts/custom/analyze_coverage.py \
    --dataset-dir ./data/verifier-dataset

# Analyze first 10 samples, summary only
python scripts/custom/analyze_coverage.py \
    --dataset-dir ./data/verifier-dataset \
    --num-samples 10 \
    --quiet
```

**Output:**
- Per-sample coverage percentages
- Summary statistics (mean, std, min, max, median) for ACCEPT, REJECT hard, REJECT easy

## Example Config

See `examples/qwen3-vl/lora-8b.yaml` for a complete config using these scripts to fine-tune Qwen3-VL-8B on MaskGroups-HQ.

## Requirements

```bash
pip install matplotlib wandb datasets tqdm pycocotools scipy
```

