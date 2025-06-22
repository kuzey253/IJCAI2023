# --- START OF FILE train.py ---

import argparse
import datetime
import torch
import numpy as np
import os
import sys
from pathlib import Path
from collections import deque, namedtuple
import random
from torch.utils.tensorboard import SummaryWriter

# --- Path Setup ---
# Assuming this script is run from the project root or base_dir is adjusted accordingly
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

from environments import make
from rl_trainer.log_path import make_logpath, save_config
from rl_trainer.algo.ppo import PPO
from utils.config import actions_map
from utils.convergence import check_convergence
from olympics_engine.generator import create_scenario

def main(args):
    print("Creating a dummy environment for initialization...")
    # Use map_id=1 for initialization
    temp_env = make(args.env, map_id=1) 
    initial_obs = temp_env.reset()
    obs_dim = initial_obs[0].flatten().shape[0]
    action_dim = len(actions_map)
    temp_env.close()

    # TODO: Set hparams according to args
    ppo_hparams = {
        'lr': 0.0003,
        'gamma': 0.99,
        'clip_param': 0.2,
        'ppo_update_time': 10,
        'buffer_capacity': 2048,
        'batch_size': 64,
        'max_grad_norm': 0.5,
    }

    ctrl_agent_index = 0
    
    # --- PATH, TENSORBOARD, AND SEED SETUP ---
    # Seeding for libraries is done once at the start
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_dir, log_dir = make_logpath(args.env, args.algo)
    
    writer = None
    if not args.load_model:
        writer = SummaryWriter(log_dir)
        save_config(args, log_dir)
    
    # --- AGENT SETUP ---
    # TODO: Set model type based on arguments
    model = PPO(run_dir=run_dir if not args.load_model else None, 
                obs_dim=obs_dim, action_dim=action_dim, **ppo_hparams)
    
    if args.load_model:
        # Adjusted path to be more robust
        load_dir = os.path.join(str(Path(run_dir).parent.parent), args.env, f"run{args.load_run}")
        model.load(load_dir, episode=args.load_episode)
        
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])
    
    training_maps = list(range(1, 11))
    # --- TRAINING LOOP ---
    episode = 0
    record_win = deque(maxlen=100)
    record_reward = deque(maxlen=100)
    
    from olympics_engine.agent import random_agent # TODO: Opponent agent should be set up based on args
    opponent_agent = random_agent()
    
    # This loop now controls the creation of environments
    while episode < args.max_episodes:
        
        map_id_to_use = None
        if args.env == 'running_competition':
            current_map_id = training_maps[episode % len(training_maps)]
            map_id_to_use = current_map_id
            # ... (rest of map cycling logic) ...
            print(f"\n--- Starting Episode {episode+1} on Map ID: {map_id_to_use} ---")

        # The make function now handles the arguments correctly for any env
        env = make(args.env, map_id=map_id_to_use)#map_id_to_use

        env.seed(args.seed + episode)
        
        state = env.reset()
        if args.render:
            env.render()
        
        obs_ctrl_agent = state[ctrl_agent_index].flatten()
        
        episode += 1
        total_reward = 0
        
        # This inner loop runs one full episode on the newly created map
        while True:
            action_ctrl_idx, action_prob = model.select_action(
                obs_ctrl_agent, train=not args.load_model)
            action_ctrl = actions_map.get(action_ctrl_idx)
            
            if env.agent_num > 1:
                action_opponent = opponent_agent.act(state[1-ctrl_agent_index])
                action = [action_ctrl, action_opponent] if ctrl_agent_index == 0 else [action_opponent, action_ctrl]
            else:
                action = [action_ctrl]

            next_state, reward, done, info = env.step(action)
            
            is_done_episode = done if isinstance(done, bool) else all(done)
            
            if not args.load_model:
                trans = Transition(obs_ctrl_agent, action_ctrl_idx, action_prob,
                                   reward[ctrl_agent_index], next_state[ctrl_agent_index].flatten(), is_done_episode)
                model.store_transition(trans)
            
            obs_ctrl_agent = next_state[ctrl_agent_index].flatten()
            state = next_state
            total_reward += reward[ctrl_agent_index]
            
            if args.render:
                env.render()

            if is_done_episode:
                # --- NEW, UNAMBIGUOUS LOGGING AND WIN CONDITION ---
                
                final_progress_ctrl_agent = info.get('final_progress', 0.0)
                win_signal = info.get('win_signal', 0)

                # A true win for OUR agent is crossing the finish line.
                # This is the only way `win_is` should be 1.
                win_is = 1 if win_signal == 1 else 0
                # If the opponent agent won, then we lost.
                if win_signal == -1:
                    win_is = -1
                
                # For logging, we determine the definitive outcome.
                outcome_msg = "Result: "
                if win_signal == 1:
                    # Agent 0 won. Since we control Agent 0, this is OUR win.
                    outcome_msg += "Controlled Agent (Team 0) WON"
                elif win_signal == -1:
                    # Agent 1 won. This is OUR loss.
                    outcome_msg += "Opponent (Team 1) WON"
                else:
                    # Neither agent won, so it was a timeout.
                    outcome_msg += "Timeout"
                
                # Now, let's record our metrics based on our controlled agent
                record_win.append(win_is) # record_win is ONLY for our agent finishing.
                record_reward.append(total_reward)
                
                # Calculate stats
                win_rate = sum(record_win) / len(record_win) if len(record_win) > 0 else 0.0
                avg_reward = sum(record_reward) / len(record_reward) if len(record_reward) > 0 else 0.0
                
                # Print the comprehensive log line
                print(f"Epi: {episode}, Env: {args.env}, R: {total_reward:.2f}, "
                      f"AvgR (last {len(record_reward)}): {avg_reward:.2f}, "
                      f"WinRate (last {len(record_win)}): {win_rate:.2f} | {outcome_msg}")

                # --- The rest of the block is unchanged ---
                if writer:
                    writer.add_scalar('metrics/total_reward', total_reward, episode)
                    writer.add_scalar('metrics/avg_reward_100', avg_reward, episode)
                    writer.add_scalar('metrics/win_rate_100', win_rate, episode)
                
                if not args.load_model and len(model.buffer) > 0:
                    model.update(episode)

                if check_convergence(episode, record_win, record_reward):
                    episode = args.max_episodes + 1
                
                break # Break inner while loop
        
        # --- NEW: Close the environment for this episode ---
        env.close()

        if episode > args.max_episodes:
            break

        if episode % args.save_interval == 0 and not args.load_model:
            print(f"\n--- Saving model at episode {episode} ---\n")
            model.save(run_dir, episode)
    
    # --- Final Save ---
    if not args.load_model:
        final_episode_num = min(episode, args.max_episodes)
        print(f"--- Saving final model from episode {final_episode_num} ---")
        model.save(run_dir, f"final_ep{final_episode_num}")

    if writer:
        writer.close()
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="curling_competition", type=str, help="Name of the environment")
    parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
    parser.add_argument('--max_episodes', default=2000, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    # Defaulting render to False is better for faster training
    parser.add_argument("--render", action='store_true', default=False, help="Render the environment during training.")
    parser.add_argument("--load_model", action='store_true', default=False)
    parser.add_argument("--load_run", default=1, type=int)
    parser.add_argument("--load_episode", default=100, type=int)
    
    args = parser.parse_args()
    # Correcting a potential path issue if script is not in the project root
    if "avoid_rl" in base_dir:
       base_dir = str(Path(__file__).resolve().parent)
       sys.path.insert(0, str(Path(base_dir).parent)) # Add project root to path
       from environments import make
    
    main(args)