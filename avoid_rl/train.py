# --- START OF FILE avoid_rl/train.py ---

import argparse
import datetime
import torch
import numpy as np
import os
import sys
from pathlib import Path
from collections import deque, namedtuple
import random
import json
from torch.utils.tensorboard import SummaryWriter

# --- Sys Path Setup ---
# This setup assumes the 'avoid_rl' folder is at the same level as 'rl_trainer'
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
# --- End Sys Path Setup ---

from rl_trainer.log_path import *
from rl_trainer.algo.ppo import PPO

from environment import LearnToAvoidEnv
from config import actions_map
from tracker import BestPathTracker
from utils import check_convergence


def main(args):
    # --- ENV SETUP ---
    env = LearnToAvoidEnv()
    obs_dim = 40 * 40
    ctrl_agent_index = 0
    
    # --- PATH, TENSORBOARD, AND SEED SETUP ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    
    if not args.load_model:
        writer = SummaryWriter(os.path.join(str(log_dir), "{}_{} on {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.algo, args.game_name)))
        save_config(args, log_dir)
        best_path_tracker = BestPathTracker(str(run_dir))
    
    # --- AGENT SETUP ---
    if args.load_model:
        model = PPO()
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir, episode=args.load_episode)
    else:
        model = PPO(run_dir)
        Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])
        
    # --- TRAINING LOOP ---
    episode = 0
    train_count = 0
    record_win = deque(maxlen=100)
    record_reward = deque(maxlen=100)
    
    while episode < args.max_episodes:
        state = env.reset()
        if args.render:
            env.render()
        
        obs_ctrl_agent = state[ctrl_agent_index].flatten()
        
        episode += 1
        step = 0
        Gt = 0
        
        while True:
            action_ctrl_raw, action_prob = model.select_action(
                obs_ctrl_agent, not args.load_model)
            
            action_ctrl = actions_map.get(action_ctrl_raw, actions_map[0]) # Fallback to a default action
            action = [action_ctrl]
            
            next_state, reward, done, info = env.step(action)
            next_obs_ctrl_agent = next_state[ctrl_agent_index].flatten()
            step += 1
            
            if not args.load_model:
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob,
                                   reward[ctrl_agent_index], next_obs_ctrl_agent, done)
                model.store_transition(trans)
            
            obs_ctrl_agent = next_obs_ctrl_agent
            Gt += reward[ctrl_agent_index]
            
            if args.render:
                env.render()
            
            if done:
                # Append results to records
                win_is = 1 if info == 'finished' else 0
                record_win.append(win_is)
                record_reward.append(Gt)

                # --- [THE FIX] ---
                # Check for convergence *immediately* after updating records for the episode.
                # This ensures we check the state *before* a potential performance collapse affects the metrics.
                if args.early_stop and not args.load_model:
                    if check_convergence(episode, record_win, record_reward):
                        print(f"Stopping training at episode {episode} due to convergence.")
                        episode = args.max_episodes  # Set to max to break outer loop
                        # We still proceed to log/save this final state before breaking.
                
                if not args.load_model:
                    # Update and save best paths/models
                    episode_data = env.get_episode_data()
                    path_updated = best_path_tracker.update(episode, Gt, episode_data)
                    
                    if path_updated:
                        stats = best_path_tracker.get_stats()
                        if Gt == stats['best_overall_reward']:
                            best_model_dir = os.path.join(str(run_dir), 'best_models')
                            os.makedirs(best_model_dir, exist_ok=True)
                            model.save(best_model_dir, f'best_overall_ep{episode}')
                        if info == 'finished' and Gt == stats['best_successful_reward']:
                            best_model_dir = os.path.join(str(run_dir), 'best_models')
                            os.makedirs(best_model_dir, exist_ok=True)
                            model.save(best_model_dir, f'best_successful_ep{episode}')

                    # Log to TensorBoard
                    writer.add_scalar('training/return', Gt, episode)
                    writer.add_scalar('training/win_rate', sum(record_win)/len(record_win), episode)
                    writer.add_scalar('training/episode_length', step, episode)

                    # Update PPO
                    if len(model.buffer) >= model.batch_size:
                        model.update(episode)
                        train_count += 1
                
                print(f"Epi: {episode}, "
                      f"Return: {Gt:.2f}, "
                      f"Win Rate: {sum(record_win)/len(record_win):.3f}, "
                      f"Avg R: {sum(record_reward)/len(record_reward):.2f}, "
                      f"Info: {info}")

                break # End inner while loop

        if episode % args.save_interval == 0 and not args.load_model:
            model.save(run_dir, episode)
            if 'best_path_tracker' in locals():
                best_path_tracker.save_periodic_summary(episode)

    if not args.load_model and 'best_path_tracker' in locals():
        print("\n" + "="*50)
        print("TRAINING COMPLETED - FINAL SUMMARY")
        print("="*50)
        final_stats = best_path_tracker.get_stats()
        final_summary_file = os.path.join(str(run_dir), 'best_paths', 'final_training_summary.json')
        # ... (rest of summary logic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="Learn2Avoid", type=str)
    parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
    parser.add_argument('--max_episodes', default=2000, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--load_run", default=1, type=int)
    parser.add_argument("--load_episode", default=100, type=int)
    parser.add_argument("--early_stop", default=True, action='store_true')
    parser.add_argument("--render", default=True, action='store_true')
    args = parser.parse_args()
    main(args)