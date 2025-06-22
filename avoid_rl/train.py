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
from olympics_engine.agent import random_agent

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
        'lr': args.lr,
        'gamma': args.gamma,
        'clip_param': args.clip_param,
        'ppo_update_time': args.ppo_update_time,
        'buffer_capacity': args.buffer_capacity,
        'batch_size': args.batch_size,
        'max_grad_norm': args.max_grad_norm,
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
    if args.algo == 'ppo':
        model = PPO(run_dir=run_dir if not args.load_model else None, 
                obs_dim=obs_dim, action_dim=action_dim, **ppo_hparams)
    elif args.algo == 'ppo-world':
        model = PPO(run_dir=run_dir if not args.load_model else None, 
                obs_dim=obs_dim, action_dim=action_dim, world=True, **ppo_hparams)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} is not implemented yet.")
    
    if args.load_model:
        # Adjusted path to be more robust
        load_dir = os.path.join(str(Path(run_dir).parent.parent), args.env, f"run{args.load_run}")
        model.load(load_dir, episode=args.load_episode)
        
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])
    
    training_maps = list(range(1, 11))
    # --- TRAINING LOOP ---
    episode = 0
    records_win = [deque(maxlen=100), deque(maxlen=100)]  # Two deques for two agents
    records_reward = [deque(maxlen=100), deque(maxlen=100)]  #

    
     # TODO: Opponent agent should be set up based on args
    if args.algo_opponent == 'ppo':
        opponent_model = PPO(run_dir=run_dir if not args.load_model else None, 
                             obs_dim=obs_dim, action_dim=action_dim, **ppo_hparams)
        if args.load_model:
            opponent_load_dir = os.path.join(str(Path(run_dir).parent.parent), args.env, f"run{args.load_run}")
            opponent_model.load(opponent_load_dir, episode=args.load_episode)
        opponent_agent = opponent_model
    elif args.algo_opponent == 'ppo-world':
        opponent_model = PPO(run_dir=run_dir if not args.load_model else None, 
                             obs_dim=obs_dim, action_dim=action_dim, world=True, **ppo_hparams)
        if args.load_model:
            opponent_load_dir = os.path.join(str(Path(run_dir).parent.parent), args.env, f"run{args.load_run}")
            opponent_model.load(opponent_load_dir, episode=args.load_episode)
        opponent_agent = opponent_model
    elif args.algo_opponent == 'random':
        # If opponent is random, we use the random_agent class
        # This should be defined in olympics_engine.agent
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
        
        # obs_ctrl_agent = state[ctrl_agent_index].flatten()
        models = [model, opponent_agent]
        obs_agents = state.copy()  # Copy the state for each agent
        
        episode += 1
        agent_total_rewards = [0, 0]
        
        # This inner loop runs one full episode on the newly created map
        while True:
            actions_res = []
            # Iterate through each agent in the environment
            for agent_index in range(env.agent_num):
                # The model selects an action based on the observation
                action_idx, action_prob = models[agent_index].select_action(
                                obs_agents[agent_index], train=not args.load_model)
                # action = actions_map.get(action_idx)
                actions_res.append((action_idx, action_prob))
            next_state, reward, done, info = env.step([actions_map.get(el[0]) for el in actions_res])
            
            is_done_episode = done if isinstance(done, bool) else all(done)
            
            
            if not args.load_model:
                for agent_index in range(env.agent_num):
                    trans = Transition(state[agent_index], actions_res[agent_index][0],
                                        actions_res[agent_index][1],
                                   reward[agent_index], next_state[agent_index], is_done_episode)
                    models[agent_index].store_transition(trans)
                obs_agents[agent_index] = next_state[agent_index]
            
                agent_total_rewards[agent_index] += reward[agent_index]

            state = next_state
            
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
                for agent_index in range(env.agent_num):
                    records_win[agent_index].append(win_is) # record_win is ONLY for our agent finishing.
                    records_reward[agent_index].append(agent_total_rewards[agent_index])
                    
                    # Calculate stats
                    win_rate = sum(records_win[agent_index]) / len(records_win[agent_index]) if len(records_win[agent_index]) > 0 else 0.0
                    avg_reward = sum(records_reward[agent_index]) / len(records_reward[agent_index]) if len(records_reward[agent_index]) > 0 else 0.0
                    
                    # Print the comprehensive log line
                    print(f"Epi: {episode}, Env: {args.env}, R: {agent_total_rewards[agent_index]:.2f}, "
                        f"AvgR (last {len(records_reward[agent_index])}): {avg_reward:.2f}, "
                        f"WinRate (last {len(records_win[agent_index])}): {win_rate:.2f} | {outcome_msg}")

                    # --- The rest of the block is unchanged ---
                    if writer:
                        writer.add_scalar(f'metrics/{agent_index}_total_reward', agent_total_rewards[agent_index], episode)
                        writer.add_scalar(f'metrics/{agent_index}_avg_reward_100', avg_reward, episode)
                        writer.add_scalar(f'metrics/{agent_index}_win_rate_100', win_rate, episode)
                    
                    if not args.load_model and len(model.buffer) > 0:
                        model.update(episode)

                    if check_convergence(episode, records_win[agent_index], records_reward[agent_index]):
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
    parser.add_argument('--algo_opponent', default="random", type=str, help="random/ppo/sac")
    parser.add_argument('--max_episodes', default=2000, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--save_interval", default=100, type=int)
    # Defaulting render to False is better for faster training
    parser.add_argument("--render", action='store_true', default=False, help="Render the environment during training.")
    parser.add_argument("--load_model", action='store_true', default=False)
    parser.add_argument("--load_run", default=1, type=int)
    parser.add_argument("--load_episode", default=100, type=int)
    # TODO: Hyperparameters for PPO
    parser.add_argument("-lr", default=0.0003, type=float, help="Learning rate for the model")
    parser.add_argument("-gamma", default=0.99, type=float, help="Discount factor for rewards")
    parser.add_argument("-clip_param", default=0.2, type=float, help="PPO clip parameter")
    parser.add_argument("-ppo_update_time", default=10, type=int, help="Number of PPO updates per episode")
    parser.add_argument("-buffer_capacity", default=2048, type=int, help="Capacity of the replay buffer")
    parser.add_argument("-batch_size", default=64, type=int, help="Batch size for training")
    parser.add_argument("-max_grad_norm", default=0.5, type=float, help="Maximum gradient norm for clipping")
    
    args = parser.parse_args()
    # Correcting a potential path issue if script is not in the project root
    if "avoid_rl" in base_dir:
       base_dir = str(Path(__file__).resolve().parent)
       sys.path.insert(0, str(Path(base_dir).parent)) # Add project root to path
       from environments import make
    
    main(args)