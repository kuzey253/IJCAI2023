# --- START OF FILE test.py ---
import os
import sys
from pathlib import Path
import torch
import argparse
import imageio
import pygame
import re

# --- Path Setup ---
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

from rl_trainer.algo.ppo import PPO
from utils.config import actions_map
from environments import make # Use the new factory

def find_latest_best_model(run_dir_base, metric):
    best_model_dir = os.path.join(run_dir_base, 'best_models')
    if not os.path.isdir(best_model_dir):
        return None, None
    
    prefix = f'actor_best_{metric}_ep'
    try:
        search_path = os.path.join(best_model_dir, 'trained_model')
        model_files = [f for f in os.listdir(search_path) if f.startswith(prefix) and f.endswith('.pth')]
    except FileNotFoundError:
        return None, None

    if not model_files:
        return None, None

    latest_ep_num = -1
    episode_to_load = None
    for f in model_files:
        match = re.search(r'ep(\d+)\.pth$', f)
        if match:
            ep_num = int(match.group(1))
            if ep_num > latest_ep_num:
                latest_ep_num = ep_num
                episode_to_load = f.replace('actor_', '').replace('.pth', '')
    
    return os.path.join(run_dir_base, 'best_models'), episode_to_load

def main(args):
    # --- ENV SETUP ---
    env = make(args.env)
    env.seed(args.seed)
    
    initial_obs = env.reset()
    obs_dim = initial_obs[0].flatten().shape[0]
    action_dim = len(actions_map)
    
    # --- AGENT SETUP ---
    model = PPO(run_dir=None, obs_dim=obs_dim, action_dim=action_dim)
    
    run_dir_base = os.path.join(base_dir, "rl_trainer", "models", args.env, f"run{args.load_run}")

    if args.load_best:
        print(f"[INFO] Finding best '{args.load_best}' model for env '{args.env}' in run {args.load_run}...")
        load_dir, episode_to_load = find_latest_best_model(run_dir_base, args.load_best)
        if not load_dir:
            sys.exit(f"\n[ERROR] Could not find any best '{args.load_best}' model in {run_dir_base}")
    else:
        load_dir = os.path.join(run_dir_base, "trained_model")
        episode_to_load = args.load_episode

    print(f"[INFO] Loading model: '{episode_to_load}' from '{load_dir}'")
    try:
        model.load(load_dir, episode=episode_to_load)
        print(f"[INFO] ✓ PPO model loaded successfully.")
    except Exception as e:
        sys.exit(f"\n[ERROR] Failed to load PPO model: {e}")

    # --- RUN EPISODE ---
    obs = env.reset()
    done = False
    frames = []
    total_reward = 0

    from olympics_engine.agent import random_agent
    opponent_agent = random_agent()

    while True:
        # obs_flat = obs[0].flatten()
        with torch.no_grad():
            action_index, _ = model.select_action(obs[0], train=False)
        
        action_ctrl = actions_map.get(action_index)
        
        if env.agent_num > 1:
            action_opponent = opponent_agent.act(obs[1])
            action = [action_ctrl, action_opponent]
        else:
            action = [action_ctrl]

        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

        env.render()
        if args.capture_gif:
            screen = pygame.display.get_surface()
            if screen:
                img = pygame.surfarray.array3d(screen).swapaxes(0, 1)
                frames.append(img)
        
        is_done_episode = done if isinstance(done, bool) else all(done)
        if is_done_episode:
            break
    
    print(f"Episode finished. Total Reward: {total_reward:.2f}. Info: {info}")
    env.close()

    # Save GIF
    if args.capture_gif and frames:
        gif_path = os.path.join(base_dir, f"{args.env}_run{args.load_run}_e{episode_to_load}.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"GIF saved successfully to {gif_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="running", type=str, help="Name of the environment to test.")
    parser.add_argument("--load_run", default=1, type=int, help="The run number to load the model from.")
    parser.add_argument("--load_episode", default=1000, type=int, help="The episode number to load.")
    parser.add_argument("--load_best", type=str, choices=['overall', 'successful'], default=None, help="Load the best model based on a metric.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--capture_gif", action='store_true', help="Capture gameplay as a GIF.")
    parser.add_argument("--use_cnn", action='store_true', help="Use CNN for the agent's network.")
    args = parser.parse_args()
    main(args)