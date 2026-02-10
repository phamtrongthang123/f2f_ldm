import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import collections

# Add diffusion_policy
sys.path.append(os.path.join(os.getcwd(), 'diffusion_policy'))

try:
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.model.common.normalizer import LinearNormalizer
except ImportError:
    print("Error: Could not import diffusion_policy.")
    sys.exit(1)

from models.policy import DriftingPolicy

def get_obs_vector(obs, agent_pos):
    # Flatten keypoints and agent_pos
    # obs['keypoint']: [9, 2] -> [18]
    # agent_pos: [2]
    # Result: [20]
    kp = obs['keypoint'].flatten()
    return np.concatenate([kp, agent_pos])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--exec_steps", type=int, default=8, help="Receding horizon execution steps")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)
    
    # Init Env
    env = PushTKeypointsEnv(render_mode='human' if args.render else None)
    
    # Init Model
    obs_dim = 20
    action_dim = 2
    policy = DriftingPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim * args.horizon, # Flattened history
        # Note: In training we flattened [B, T, Do]. 
        # Here we need to match that.
        # If training used T=16 obs history, we need to maintain history.
        horizon=args.horizon,
        embed_dim=256,
        depth=6,
        num_heads=8
    ).to(args.device)
    
    policy.load_state_dict(ckpt['model'])
    policy.eval()
    
    # Load Normalizer
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt['normalizer'])
    
    total_rewards = []
    success_count = 0
    
    for i in tqdm(range(args.n_episodes)):
        obs = env.reset()
        # Create obs history buffer
        # In training, we used SequenceSampler which gives T steps of obs.
        # Here we need to stack current obs T times or wait for history?
        # Standard DP approach: pad with first obs.
        
        # We need a queue of observations
        obs_deque = collections.deque([get_obs_vector(obs, env._get_info()['pos_agent'])] * args.horizon, maxlen=args.horizon)
        
        done = False
        step_count = 0
        rewards = 0
        
        while not done:
            # 1. Prepare Observation
            # Stack deque -> [T, Do]
            obs_seq = np.stack(obs_deque) # [T, 20]
            # Normalize
            nobs = normalizer.normalize(obs_seq, key='obs')
            
            # Batchify
            nobs_tensor = torch.from_numpy(nobs).unsqueeze(0).float().to(args.device) # [1, T, 20]
            
            # Flatten for model cond
            cond = nobs_tensor.reshape(1, -1)
            
            # 2. Generate Action
            with torch.no_grad():
                noise = torch.randn(1, args.horizon, action_dim, device=args.device)
                naction = policy(noise, cond) # [1, T, 2]
                
            # Unnormalize
            naction = naction.detach().cpu().numpy()[0] # [T, 2]
            action = normalizer['action'].unnormalize(naction)
            
            # 3. Execute
            # Receding horizon: execute first k steps
            start = step_count
            end = min(step_count + args.exec_steps, 300) # max steps
            
            for j in range(args.exec_steps):
                if j >= len(action): break
                
                act = action[j]
                obs, reward, done, info = env.step(act)
                rewards += reward
                step_count += 1
                
                # Update obs buffer
                obs_vec = get_obs_vector(obs, info['pos_agent'])
                obs_deque.append(obs_vec)
                
                if args.render:
                    env.render()
                    
                if done:
                    break
            
            if step_count >= 300:
                done = True
                
        total_rewards.append(rewards)
        if rewards >= 1.0: # Approx max reward for success
            success_count += 1
            
    print(f"Success Rate: {success_count/args.n_episodes:.2f}")
    print(f"Avg Reward: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    main()
