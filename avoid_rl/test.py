# --- START OF FILE avoid_rl/test.py ---

import os
import sys
from pathlib import Path
import torch
import argparse
import re
import imageio
import pygame

# --- Sys Path Setup ---
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
# --- End Sys Path Setup ---

from rl_trainer.algo.ppo import PPO
from environment import LearnToAvoidEnv
from config import actions_map

def main(args):
    # --- ENV SETUP ---
    env = LearnToAvoidEnv()
    
    # --- AGENT SETUP ---
    model = PPO()
    
    # Build path to the specified run
    run_dir_base = os.path.join(
        base_dir, "rl_trainer", "models", args.game_name, f"run{args.load_run}"
    )

    load_dir = run_dir_base
    episode_to_load = args.load_episode

    if args.load_best:
        print(f"[INFO] Finding best '{args.load_best}' model...")
        best_model_dir = os.path.join(run_dir_base, 'best_models')
        if not os.path.isdir(best_model_dir):
            sys.exit(f"\n[ERROR] 'best_models' dir not found in {run_dir_base}")

        load_dir = best_model_dir
        search_path = os.path.join(load_dir, 'trained_model')
        prefix = f'actor_best_{args.load_best}_ep'
        
        try:
            model_files = [f for f in os.listdir(search_path) if f.startswith(prefix) and f.endswith('.pth')]
        except FileNotFoundError:
            sys.exit(f"\n[ERROR] Could not find '{search_path}'.")

        if not model_files:
            sys.exit(f"\n[ERROR] No best '{args.load_best}' models found in {search_path}")

        # Find the model with the highest episode number
        latest_ep_num = -1
        for f in model_files:
            match = re.search(r'ep(\d+)\.pth$', f)
            if match:
                ep_num = int(match.group(1))
                if ep_num > latest_ep_num:
                    latest_ep_num = ep_num
                    episode_to_load = f.replace('actor_', '').replace('.pth', '')
        
        if latest_ep_num == -1:
            sys.exit(f"\n[ERROR] Could not parse episode numbers from {search_path}")

        print(f"[INFO] Found latest best model: '{episode_to_load}'")
    else:
        print(f"[INFO] Loading model from episode: {args.load_episode}")

    # Load the specified model
    try:
        model.load(load_dir, episode=episode_to_load)
        print(f"[INFO] ✓ PPO successfully loaded from: {load_dir}")
    except Exception as e:
        sys.exit(f"\n[ERROR] Failed to load PPO model: {e}")

    # --- RUN EPISODE ---
    obs = env.reset()
    done = False
    frames = []

    while not done:
        obs_flat = obs[0].flatten()
        with torch.no_grad():
            # --- [THE FIX] ---
            # Incorrect: action_index, _ = model.select_action(obs_flat, explore=False)
            # Correct: Call with a positional boolean, just like in train.py
            action_index, _ = model.select_action(obs_flat, False)
        
        action = [actions_map[action_index]]
        obs, reward, done, info = env.step(action)

        env.render()
        if args.capture_gif:
            screen = pygame.display.get_surface()
            if screen:
                img = pygame.surfarray.array3d(screen).swapaxes(0, 1)
                frames.append(img)
    
    print(f"Episode finished. Info: {info}")

    # Save GIF
    if args.capture_gif and frames:
        gif_path = os.path.join(base_dir, f"{args.game_name}_test_{args.load_run}.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"GIF saved successfully to {gif_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", default="Learn2Avoid", type=str)
    parser.add_argument("--load_run", default=1, type=int)
    parser.add_argument("--load_episode", default=100, type=int, help="Load a specific episode (overridden by --load_best).")
    parser.add_argument("--load_best", type=str, choices=['overall', 'successful'], default=None, help="Load the 'overall' or 'successful' best model.")
    parser.add_argument("--capture_gif", action='store_true', help="Capture gameplay as a GIF.")
    args = parser.parse_args()
    main(args)

    