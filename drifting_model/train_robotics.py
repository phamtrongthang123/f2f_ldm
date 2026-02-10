import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

# Add diffusion_policy to path
sys.path.append(os.path.join(os.getcwd(), 'diffusion_policy'))

try:
    from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
    from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
    from diffusion_policy.dataset.kitchen_lowdim_dataset import KitchenLowdimDataset
    from diffusion_policy.model.common.normalizer import LinearNormalizer
except ImportError as e:
    print(f"Error: Could not import diffusion_policy modules: {e}")
    sys.exit(1)

from models.policy import DriftingPolicy
from drifting_loss import compute_drifting_loss

def get_dataset(args):
    horizon = args.horizon
    if args.task == 'pusht':
        return PushTLowdimDataset(
            zarr_path=args.data_dir,
            horizon=horizon,
            pad_before=args.pad_before,
            pad_after=args.pad_after,
            obs_key='keypoint',
            state_key='state',
            action_key='action'
        )
    elif args.task in ['lift', 'can', 'square']:
        # Map task name to dataset path if not provided explicitly
        if args.data_dir == "data/pusht_cchi_v7_replay.zarr": # Default was pusht
             # Try to guess
             guessed_path = f"data/robomimic_{args.task}_ph.hdf5"
             if os.path.exists(guessed_path):
                 args.data_dir = guessed_path
             else:
                 # Check for folder structure from zip
                 guessed_path = f"data/{args.task}_ph/low_dim.hdf5" # Common structure
                 if not os.path.exists(guessed_path):
                     # Fallback to just using what user provided or error
                     pass
        
        return RobomimicReplayLowdimDataset(
            dataset_path=args.data_dir,
            horizon=horizon,
            pad_before=args.pad_before,
            pad_after=args.pad_after,
            obs_keys=['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        )
    elif args.task == 'kitchen':
        return KitchenLowdimDataset(
            dataset_dir=args.data_dir,
            horizon=horizon,
            pad_before=args.pad_before,
            pad_after=args.pad_after
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")

def main():
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pusht", choices=["pusht", "lift", "can", "square", "kitchen"])
    parser.add_argument("--data_dir", type=str, default="data/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3050)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints/robotics")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--pad_before", type=int, default=1)
    parser.add_argument("--pad_after", type=int, default=7)
    args = parser.parse_args()

    # Update save dir with task
    args.save_dir = os.path.join(args.save_dir, args.task)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading dataset for task: {args.task}...")
    try:
        dataset = get_dataset(args)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please check data_dir and ensure data is downloaded via download_data.sh")
        return

    # Determine dims from a sample
    sample = dataset[0]
    # sample['obs'] is [T, Do], sample['action'] is [T, Da]
    obs_dim = sample['obs'].shape[-1]
    action_dim = sample['action'].shape[-1]
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Normalizer
    normalizer = dataset.get_normalizer()
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # 2. Model
    # Condition on flattened history of n_obs_steps
    cond_dim = obs_dim * args.n_obs_steps
    
    policy = DriftingPolicy(
        action_dim=action_dim,
        obs_dim=cond_dim, 
        horizon=args.horizon,
        embed_dim=256,
        depth=6,
        num_heads=8
    ).to(args.device)
    
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Drifting Loss Params
    temperatures = [0.02, 0.05, 0.2]
    
    # 3. Training Loop
    print(f"Starting training on {args.device}...")
    
    for epoch in range(args.epochs):
        policy.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        
        for batch in pbar:
            nobs = batch['obs'].to(args.device)
            naction = batch['action'].to(args.device)
            
            # Normalize
            nobs = normalizer['obs'].normalize(nobs)
            naction = normalizer['action'].normalize(naction)
            
            B = nobs.shape[0]
            
            # Flatten obs for condition: take first n_obs_steps
            cond = nobs[:, :args.n_obs_steps, :].reshape(B, -1)
            
            # Target action (flattened)
            y_pos = naction.reshape(B, -1)
            
            # Generate
            noise = torch.randn(B, args.horizon, action_dim, device=args.device)
            gen_action = policy(noise, cond)
            x = gen_action.reshape(B, -1)
            
            loss = compute_drifting_loss(
                gen_features=[], pos_features=[], neg_features=[], uncond_features=[],
                temperatures=temperatures,
                cfg_weights=None,
                gen_latent=x,
                pos_latent=y_pos,
                neg_latent=x,
                uncond_latent=None
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader):.4f}")
        
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'model': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'epoch': epoch,
                'args': args
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()