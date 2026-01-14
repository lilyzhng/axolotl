"""
Custom W&B plugin for logging images and predictions during VLM training.

This plugin logs:
1. Sample input images from the dataset
2. Model predictions vs ground truth at each eval step

Usage in axolotl config:
  plugins:
    - scripts.wandb_image_callback.WandbImagePlugin

Make sure PYTHONPATH includes /workspace/fine-tuning when running:
  PYTHONPATH=/workspace/fine-tuning accelerate launch -m axolotl.cli.train ...
"""

import random
from typing import Optional
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments, Trainer
from PIL import Image

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import BasePlugin - handle both installed and development axolotl
try:
    from axolotl.integrations.base import BasePlugin
    from axolotl.utils.dict import DictDefault
except ImportError:
    # Fallback for when axolotl is not properly installed
    BasePlugin = object
    DictDefault = dict


class WandbImageCallback(TrainerCallback):
    """
    Callback to log images and predictions to Weights & Biases.
    
    Args:
        num_samples: Number of samples to log per evaluation (default: 4)
        log_predictions: Whether to generate and log model predictions (default: True)
        max_new_tokens: Max tokens to generate for predictions (default: 64)
    """
    
    def __init__(
        self,
        num_samples: int = 4,
        log_predictions: bool = True,
        max_new_tokens: int = 64,
    ):
        self.num_samples = num_samples
        self.log_predictions = log_predictions
        self.max_new_tokens = max_new_tokens
        self.eval_dataset = None
        self.processor = None
        self.sample_indices = None
        self._logged_initial = False
        
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize with eval dataset and processor at training start."""
        if not WANDB_AVAILABLE or wandb.run is None:
            print("[WandbImageCallback] W&B not available or not initialized")
            return
            
        # Get eval dataset and processor from trainer
        self.eval_dataset = kwargs.get("eval_dataset")
        self.processor = kwargs.get("processing_class") or kwargs.get("tokenizer")
        
        if self.eval_dataset is not None and not self._logged_initial:
            # Select random samples to track throughout training
            dataset_size = len(self.eval_dataset)
            self.sample_indices = random.sample(
                range(dataset_size), 
                min(self.num_samples, dataset_size)
            )
            print(f"[WandbImageCallback] Tracking {len(self.sample_indices)} samples for visualization")
            
            # Log initial sample images
            self._log_sample_images()
            self._logged_initial = True
    
    def _log_sample_images(self):
        """Log the selected sample images to W&B."""
        if self.eval_dataset is None or wandb.run is None:
            return
            
        images_to_log = []
        
        for idx in self.sample_indices:
            try:
                sample = self.eval_dataset[idx]
                
                # Try to extract image from sample
                image = self._extract_image(sample)
                if image is not None:
                    # Get the text prompt if available
                    caption = self._extract_prompt(sample)
                    ground_truth = self._extract_ground_truth(sample)
                    images_to_log.append(
                        wandb.Image(
                            image, 
                            caption=f"Sample {idx}\nPrompt: {caption[:80]}...\nGT: {ground_truth[:80]}..."
                        )
                    )
            except Exception as e:
                print(f"[WandbImageCallback] Error processing sample {idx}: {e}")
        
        if images_to_log:
            wandb.log({"samples/input_images": images_to_log}, commit=False)
            print(f"[WandbImageCallback] Logged {len(images_to_log)} sample images")
    
    def _extract_image(self, sample) -> Optional[Image.Image]:
        """Extract PIL Image from a dataset sample."""
        # Check common image field names
        for key in ["images", "image", "pixel_values"]:
            if key not in sample:
                continue
                
            img_data = sample[key]
            
            # Handle list of images
            if isinstance(img_data, list) and len(img_data) > 0:
                img_data = img_data[0]
            
            # Handle PIL Image
            if isinstance(img_data, Image.Image):
                return img_data
            
            # Handle file path
            if isinstance(img_data, str):
                try:
                    return Image.open(img_data).convert("RGB")
                except Exception:
                    pass
            
            # Handle tensor
            if isinstance(img_data, torch.Tensor):
                try:
                    # Assume CHW format, normalize to 0-255
                    if img_data.dim() == 3:
                        img_np = img_data.permute(1, 2, 0).cpu().numpy()
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype("uint8")
                        return Image.fromarray(img_np)
                except Exception:
                    pass
        
        return None
    
    def _extract_prompt(self, sample) -> str:
        """Extract text prompt from a dataset sample."""
        # Check for messages format
        if "messages" in sample:
            messages = sample["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                first_msg = messages[0]
                if isinstance(first_msg, dict) and "content" in first_msg:
                    content = first_msg["content"]
                    # Handle multi-content format
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                return item.get("text", "")
                    return str(content)
        
        # Check for conversations format
        if "conversations" in sample:
            convs = sample["conversations"]
            if isinstance(convs, list) and len(convs) > 0:
                return str(convs[0].get("value", ""))
        
        # Check for direct prompt field
        for key in ["prompt", "instruction", "query", "question"]:
            if key in sample:
                return str(sample[key])
        
        return "No prompt found"
    
    def _extract_ground_truth(self, sample) -> str:
        """Extract ground truth response from sample."""
        if "messages" in sample:
            messages = sample["messages"]
            # Find assistant message
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                return item.get("text", "")
                    return str(content)
        
        if "conversations" in sample:
            convs = sample["conversations"]
            for conv in convs:
                if conv.get("from") in ["gpt", "assistant"]:
                    return str(conv.get("value", ""))
        
        for key in ["response", "output", "answer", "completion"]:
            if key in sample:
                return str(sample[key])
        
        return "No ground truth found"
    
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Log predictions during evaluation."""
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        
        if not self.log_predictions or model is None:
            return
            
        if self.eval_dataset is None or self.processor is None:
            return
        
        if self.sample_indices is None:
            return
        
        # Generate predictions for tracked samples
        predictions_table = self._generate_predictions(model, state.global_step)
        
        if predictions_table is not None:
            wandb.log({
                "samples/predictions": predictions_table,
            })
            print(f"[WandbImageCallback] Logged predictions table at step {state.global_step}")
    
    def _generate_predictions(self, model, step: int):
        """Generate model predictions for tracked samples."""
        if self.sample_indices is None:
            return None
            
        try:
            model.eval()
            rows = []
            
            for idx in self.sample_indices[:2]:  # Limit to 2 samples per eval for speed
                try:
                    sample = self.eval_dataset[idx]
                    
                    # Extract image and prompt
                    image = self._extract_image(sample)
                    prompt = self._extract_prompt(sample)
                    
                    if image is None:
                        continue
                    
                    # Get ground truth
                    ground_truth = self._extract_ground_truth(sample)
                    
                    # Generate prediction
                    try:
                        prediction = self._run_inference(model, image, prompt)
                    except Exception as e:
                        prediction = f"[Error: {str(e)[:50]}]"
                    
                    rows.append([
                        step,
                        wandb.Image(image),
                        prompt[:150] + "..." if len(prompt) > 150 else prompt,
                        ground_truth[:150] + "..." if len(ground_truth) > 150 else ground_truth,
                        prediction[:150] + "..." if len(prediction) > 150 else prediction,
                    ])
                except Exception as e:
                    print(f"[WandbImageCallback] Error processing sample {idx}: {e}")
            
            if rows:
                return wandb.Table(
                    columns=["Step", "Image", "Prompt", "Ground Truth", "Prediction"],
                    data=rows
                )
        except Exception as e:
            print(f"[WandbImageCallback] Error generating predictions: {e}")
        
        return None
    
    def _run_inference(self, model, image: Image.Image, prompt: str) -> str:
        """Run inference on a single image-prompt pair."""
        if self.processor is None:
            return "[No processor available]"
        
        try:
            # Clean prompt - remove <image> placeholder
            clean_prompt = prompt.replace("<image>", "").strip()
            
            # Prepare inputs using the processor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": clean_prompt},
                    ],
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            
            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            
            # Decode
            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            prediction = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return prediction.strip()
            
        except Exception as e:
            return f"[Inference error: {str(e)[:100]}]"


class WandbImagePlugin(BasePlugin):
    """
    Axolotl plugin that adds W&B image logging callback.
    
    Usage in config:
        plugins:
          - scripts.wandb_image_callback.WandbImagePlugin
    """
    
    def __init__(self):
        super().__init__()
        self.callback = None
    
    def register(self, cfg: dict):
        """Register the plugin with config."""
        print("[WandbImagePlugin] Registered for image logging")
    
    def add_callbacks_post_trainer(self, cfg, trainer: Trainer) -> list:
        """Add the image logging callback after trainer is created."""
        if not WANDB_AVAILABLE:
            print("[WandbImagePlugin] W&B not available, skipping image callback")
            return []
        
        # Create callback with access to trainer's eval dataset and processor
        self.callback = WandbImageCallback(
            num_samples=4,
            log_predictions=True,
            max_new_tokens=64,
        )
        
        # Pre-populate with trainer's data
        if hasattr(trainer, 'eval_dataset') and trainer.eval_dataset is not None:
            self.callback.eval_dataset = trainer.eval_dataset
            print(f"[WandbImagePlugin] Found eval_dataset with {len(trainer.eval_dataset)} samples")
        else:
            print("[WandbImagePlugin] Warning: No eval_dataset found in trainer")
            
        if hasattr(trainer, 'processing_class') and trainer.processing_class is not None:
            self.callback.processor = trainer.processing_class
            print("[WandbImagePlugin] Using processing_class as processor")
        elif hasattr(trainer, 'tokenizer') and trainer.tokenizer is not None:
            self.callback.processor = trainer.tokenizer
            print("[WandbImagePlugin] Using tokenizer as processor")
        else:
            print("[WandbImagePlugin] Warning: No processor found in trainer")
        
        # Initialize sample indices and log initial images now
        if self.callback.eval_dataset is not None and not self.callback._logged_initial:
            dataset_size = len(self.callback.eval_dataset)
            self.callback.sample_indices = random.sample(
                range(dataset_size), 
                min(self.callback.num_samples, dataset_size)
            )
            print(f"[WandbImagePlugin] Tracking {len(self.callback.sample_indices)} samples for visualization")
            
            # Log initial sample images if W&B is initialized
            if wandb.run is not None:
                self.callback._log_sample_images()
                self.callback._logged_initial = True
        
        print("[WandbImagePlugin] Added WandbImageCallback")
        return [self.callback]
