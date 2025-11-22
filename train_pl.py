#!/usr/bin/env python3
"""
PyTorch Lightning version of F2FLDM training.
Simplified training without Accelerate - uses pure PyTorch Lightning.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModelDev,
)
from diffusers.models.lora import LoRALinearLayer
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from feature_extractor_ultrasound import get_feat_model, get_transform


class UltrasoundDataset(Dataset):
    """Dataset for ultrasound images with DINO features."""

    def __init__(self, metadata_file, image_dir, resolution=512,
                 image_transform=None, feat_transform=None, extract_features=True):
        self.df = pd.read_csv(metadata_file)
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.extract_features = extract_features
        self.feat_transform = feat_transform

        # Don't store model - load it per worker to avoid pickling issues
        self.feat_model = None

        # Image preprocessing
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lazy load feature model in worker (avoids pickling issues with spawn)
        if self.extract_features and self.feat_model is None:
            from feature_extractor_ultrasound import get_feat_model
            self.feat_model, _ = get_feat_model("dinov2")
            self.feat_model.eval()
            if torch.cuda.is_available():
                self.feat_model = self.feat_model.cuda()

        row = self.df.iloc[idx]

        # Load image
        img_path = self.image_dir / row['image'].split('_')[0] / row['image']
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.image_transform(image)

        # Extract DINO features
        if self.extract_features and self.feat_model is not None:
            with torch.no_grad():
                feat_input = self.feat_transform(image).unsqueeze(0)
                if torch.cuda.is_available():
                    feat_input = feat_input.cuda()
                features = self.feat_model(feat_input).squeeze(0).cpu()
        else:
            features = torch.zeros(384)  # Placeholder

        return {
            'pixel_values': pixel_values,
            'features': features,
            'caption': row['text']
        }


class SDXLLoRAModule(pl.LightningModule):
    """PyTorch Lightning module for SDXL LoRA fine-tuning."""

    def __init__(
        self,
        pretrained_model_path="stabilityai/stable-diffusion-xl-base-1.0",
        lora_rank=8,
        learning_rate=1e-4,
        unet_addition_embed_type="text_latent_addembeddingft",
        train_text_encoder=True,
        resolution=512,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_path, subfolder="scheduler"
        )

        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae"
        )

        self.unet = UNet2DConditionModelDev.from_pretrained(
            pretrained_model_path,
            subfolder="unet",
            addition_embed_type=unet_addition_embed_type,
            low_cpu_mem_usage=False
        )

        self.tokenizer_one = AutoTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer", use_fast=False
        )
        self.tokenizer_two = AutoTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer_2", use_fast=False
        )

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_path, subfolder="text_encoder_2"
        )

        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Reset and enable custom embeddings
        print("Resetting UNet add_embedding...")
        self.unet.reset_add_embedding(4096, self.unet.time_embed_dim)

        if "latent" in unet_addition_embed_type:
            self.unet.latent_proj.requires_grad_(True)
            self.unet.add_embedding.requires_grad_(True)

        # Add LoRA layers
        self._setup_lora(lora_rank)

        # Training settings
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.automatic_optimization = True

    def _setup_lora(self, rank):
        """Add LoRA layers to UNet attention."""
        self.lora_parameters = []

        for attn_processor_name, attn_processor in self.unet.attn_processors.items():
            attn_module = self.unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            # Add LoRA to Q, K, V, Out
            for layer_name in ['to_q', 'to_k', 'to_v']:
                layer = getattr(attn_module, layer_name)
                lora_layer = LoRALinearLayer(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    rank=rank
                )
                layer.set_lora_layer(lora_layer)
                self.lora_parameters.extend(lora_layer.parameters())

            # to_out is a ModuleList
            out_layer = attn_module.to_out[0]
            lora_layer = LoRALinearLayer(
                in_features=out_layer.in_features,
                out_features=out_layer.out_features,
                rank=rank
            )
            out_layer.set_lora_layer(lora_layer)
            self.lora_parameters.extend(lora_layer.parameters())

    def encode_prompt(self, captions):
        """Encode text prompts to embeddings."""
        # Tokenize
        tokens_one = self.tokenizer_one(
            captions, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)

        tokens_two = self.tokenizer_two(
            captions, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)

        # Encode
        with torch.no_grad():
            enc_one = self.text_encoder_one(tokens_one, output_hidden_states=True)
            enc_two = self.text_encoder_two(tokens_two, output_hidden_states=True)

            pooled_embeds = enc_two[0]
            prompt_embeds = torch.cat([
                enc_one.hidden_states[-2],
                enc_two.hidden_states[-2]
            ], dim=-1)

        return prompt_embeds, pooled_embeds

    def forward(self, batch):
        """Forward pass - compute loss."""
        pixel_values = batch['pixel_values']
        features = batch['features'].to(self.device)
        captions = batch['caption']

        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=latents.device
        ).long()

        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode prompts
        prompt_embeds, pooled_embeds = self.encode_prompt(captions)

        # Prepare added_cond_kwargs for SDXL
        # Time IDs: [original_height, original_width, crop_top, crop_left, target_height, target_width]
        # Use actual resolution (512 for ultrasound, 1024 for default SDXL)
        res = self.resolution
        add_time_ids = torch.tensor([
            [res, res, 0, 0, res, res]
        ]).repeat(bsz, 1).to(self.device, dtype=self.dtype)

        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": add_time_ids,
            "latent_embeds": features  # Our DINO features
        }

        # Predict noise
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=added_cond_kwargs
        ).sample

        # Compute loss
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Collect all trainable parameters
        params = []
        params.extend(self.lora_parameters)

        if hasattr(self.unet, 'latent_proj') and self.unet.latent_proj is not None:
            params.extend(self.unet.latent_proj.parameters())
        if hasattr(self.unet, 'add_embedding') and self.unet.add_embedding is not None:
            params.extend(self.unet.add_embedding.parameters())

        optimizer = torch.optim.AdamW(params, lr=self.learning_rate)

        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Save only LoRA weights and custom layers."""
        # Save LoRA state dict
        lora_state = {}
        for name, module in self.unet.named_modules():
            if hasattr(module, 'lora_layer'):
                lora_state[name] = module.lora_layer.state_dict()

        checkpoint['lora_state_dict'] = lora_state

        # Save custom embedding layers
        if hasattr(self.unet, 'latent_proj') and self.unet.latent_proj is not None:
            checkpoint['latent_proj'] = self.unet.latent_proj.state_dict()
        if hasattr(self.unet, 'add_embedding') and self.unet.add_embedding is not None:
            checkpoint['add_embedding'] = self.unet.add_embedding.state_dict()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                       default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/pl_output")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed",
                       choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--val_check_interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unet_addition_embed_type", type=str,
                       default="text_latent_addembeddingft")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Set seed
    pl.seed_everything(args.seed)

    # Get feature transform (model will be loaded per-worker to avoid pickling)
    print("Setting up dataset...")
    feat_transform = get_transform("dinov2", is_ultrasound=True)

    # Create dataset (features extracted lazily in workers)
    print("Loading dataset...")
    dataset = UltrasoundDataset(
        metadata_file=args.train_data_dir,
        image_dir=args.dataset_dir,
        resolution=args.resolution,
        feat_transform=feat_transform,
        extract_features=True
    )

    # Use spawn method with workers for faster data loading
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if args.num_workers > 0 else 0,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False,
        multiprocessing_context='spawn' if args.num_workers > 0 else None
    )

    # Create model
    print("Initializing model...")
    model = SDXLLoRAModule(
        pretrained_model_path=args.pretrained_model_name_or_path,
        lora_rank=args.rank,
        learning_rate=args.learning_rate,
        unet_addition_embed_type=args.unet_addition_embed_type,
        resolution=args.resolution
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='checkpoint-{step:06d}',
        every_n_train_steps=args.val_check_interval,
        save_top_k=-1,  # Save all checkpoints
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='lightning_logs'
    )

    # Trainer (single GPU, no DDP)
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_steps=args.max_steps,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=1,  # Single GPU only
        strategy='auto',  # No DDP for single GPU
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        num_sanity_val_steps=0,  # Skip validation sanity check
    )

    # Train
    print("Starting training...")
    trainer.fit(model, dataloader)

    print(f"✓ Training complete! Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
