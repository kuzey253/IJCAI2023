# Improved RL training script with better reward shaping and exploration

import argparse
import datetime
import math

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)

from collections import deque, namedtuple
import random
import json
import pickle

from olympics_engine.generator import create_scenario
from env.chooseenv import make
from rl_trainer.log_path import *
from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.random import random_agent

from olympics_engine.core import OlympicsBase
from olympics_engine.objects import *
from olympics_engine.viewer import Viewer, debug

import pygame
import time

gamemap = {'objects':[], 'agents':[]}

gamemap['objects'].append(Wall(init_pos=[[50, 250], [650, 250]], length = None, color = 'black'))
gamemap['objects'].append(Wall(init_pos=[[50, 450], [650, 450]], length = None, color = 'black'))
gamemap['objects'].append(Wall(init_pos=[[50, 250], [50, 450]], length = None, color = 'black'))

gamemap['objects'].append(Cross(init_pos=[[650, 250], [650, 450]], length = None, color = 'red', width = 5))
gamemap['objects'].append(Cross(init_pos=[[200, 250], [200, 300]], length = None, color = 'green', width = 5))
gamemap['objects'].append(Cross(init_pos=[[250, 450], [250, 400]], length = None, color = 'green', width = 5))
gamemap['objects'].append(Cross(init_pos=[[300, 300], [300, 350]], length = None, color = 'green', width = 5))
gamemap['objects'].append(Cross(init_pos=[[450, 250], [450, 350]], length = None, color = 'green', width = 5))
gamemap['objects'].append(Cross(init_pos=[[550, 450], [550, 350]], length = None, color = 'green', width = 5))

gamemap['agents'].append(Agent(position = [75,300], mass=1, r=15, color='light red', vis_clear=5, vis=200))
gamemap['view'] = {'width': 600, 'height':600, 'edge': 50, "init_obs": [0]}

def point2point(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class env_test(OlympicsBase):
    def __init__(self, map=gamemap, point_left_prob = 1):
        # Initialize tracking variables BEFORE calling super().__init__
        self.map = map
        self.gamma = 1
        self.wall_restitution = 0.5
        self.print_log = False
        self.tau = 0.1
        self.max_step = 500  # Increased from 300
        
        self.draw_obs = True
        self.show_traj = False
        
        self.cross_color = 'red'
        self.penalty_color = 'green'
        
        # Store initial position for distance calculation
        self.initial_pos = [75, 300]
        self.goal_pos = [650, 350]  # Approximate goal position
        
        # Track previous position for movement reward
        self.prev_pos = None
        self.stuck_counter = 0
        self.max_stuck_steps = 30  # Increased tolerance
        
        # Track obstacle collisions more intelligently
        self.obstacle_collision_count = 0
        self.recent_positions = deque(maxlen=20)  # Track more positions for better exploration
        
        # Progress tracking - define key waypoints/checkpoints
        self.checkpoints = [
            [150, 350],  # Past first obstacle
            [275, 350],  # Past second obstacle  
            [375, 325],  # Past third obstacle
            [500, 300],  # Past fourth obstacle
            [575, 375],  # Past fifth obstacle
            [650, 350]   # Goal
        ]
        self.current_checkpoint = 0
        self.checkpoint_reached = [False] * len(self.checkpoints)
        
        # Track best progress achieved
        self.best_x_progress = 75
        self.stagnation_counter = 0
        
        # Path tracking for best trajectory recording
        self.current_path = []
        self.episode_actions = []
        
        # Now call parent init (which calls reset())
        super(env_test, self).__init__(map)
        
        finals = []
        penalty = []
        for object_idx in range(len(self.map['objects'])):
            object = self.map['objects'][object_idx]
            if object.can_pass():
                if object.color == self.cross_color:
                    finals.append(object)
                elif object.color == self.penalty_color:
                    penalty.append(object)
        self.finals = finals
        self.penalty = penalty
        self.info = ''

    def check_overlap(self):
        pass

    def check_action(self, action_list):
        action = []
        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].type == 'agent':
                action.append(action_list[0])
                _ = action_list.pop(0)
            else:
                action.append(None)
        return action

    def step(self, actions_list):
        previous_pos = self.agent_pos.copy()
        actions_list = self.check_action(actions_list)
        
        # Record position and action for path tracking
        self.current_path.append([self.agent_pos[0][0], self.agent_pos[0][1]])
        if len(actions_list) > 0 and actions_list[0] is not None:
            self.episode_actions.append(actions_list[0])
        
        self.stepPhysics(actions_list, self.step_cnt)
        self.cross_detect(self.agent_pos)
        
        self.step_cnt += 1
        step_reward = self.get_reward(previous_pos)
        obs_next = self.get_obs()
        done = self.is_terminal()
        
        return obs_next, step_reward, done, self.info

    def get_reward(self, previous_pos):
        agent_reward = [0. for _ in range(self.agent_num)]
        
        for agent_idx in range(self.agent_num):
            current_pos = self.agent_pos[agent_idx]
            
            # Terminal rewards
            if self.agent_list[agent_idx].finished:
                if self.agent_list[agent_idx].color == self.cross_color:
                    agent_reward[agent_idx] = 200.0  # Increased success reward
                elif self.agent_list[agent_idx].color == self.penalty_color:
                    agent_reward[agent_idx] = -50.0  # Reduced collision penalty
                    self.obstacle_collision_count += 1
            else:
                # Base time penalty (smaller)
                agent_reward[agent_idx] -= 0.01
                
                # Checkpoint-based progress reward
                checkpoint_reward = self.get_checkpoint_reward(current_pos)
                agent_reward[agent_idx] += checkpoint_reward
                
                # Distance-based reward (more nuanced)
                current_dist = point2point(current_pos, self.goal_pos)
                prev_dist = point2point(previous_pos[agent_idx], self.goal_pos)
                
                # Progressive distance reward - stronger when closer to goal
                if current_dist < prev_dist:
                    distance_factor = max(0.1, 1.0 - (current_dist / 600))  # Stronger when closer
                    distance_reward = (prev_dist - current_dist) * distance_factor * 0.2
                    agent_reward[agent_idx] += distance_reward
                
                # Smart obstacle avoidance - only penalize if very close AND not making progress
                min_obstacle_dist = self.get_min_obstacle_distance(current_pos)
                if min_obstacle_dist < 20:  # Very close to obstacle
                    # Only penalize if not making forward progress
                    x_progress = current_pos[0] - previous_pos[agent_idx][0]
                    if x_progress <= 0:  # Not making forward progress
                        agent_reward[agent_idx] -= 1.0
                elif min_obstacle_dist < 35:  # Moderately close
                    # Small penalty only if moving toward obstacle
                    if self.is_moving_toward_obstacle(current_pos, previous_pos[agent_idx]):
                        agent_reward[agent_idx] -= 0.2
                
                # Forward progress reward (encouraging rightward movement)
                x_progress = current_pos[0] - previous_pos[agent_idx][0]
                if x_progress > 0:
                    # Scale reward based on current x position (more reward for later progress)
                    progress_multiplier = 1 + (current_pos[0] - 75) / 575  # 1.0 to 2.0
                    agent_reward[agent_idx] += x_progress * 0.05 * progress_multiplier
                
                # Exploration reward - encourage visiting new areas
                exploration_reward = self.get_exploration_reward(current_pos)
                agent_reward[agent_idx] += exploration_reward
                
                # Anti-stagnation mechanism
                if current_pos[0] > self.best_x_progress:
                    self.best_x_progress = current_pos[0]
                    self.stagnation_counter = 0
                    agent_reward[agent_idx] += 2.0  # Reward for new best progress
                else:
                    self.stagnation_counter += 1
                    if self.stagnation_counter > 50:  # Stagnating too long
                        agent_reward[agent_idx] -= 0.5
                
                # Wall collision penalty (reduced)
                if self.check_wall_collision(current_pos, previous_pos[agent_idx]):
                    agent_reward[agent_idx] -= 5.0
                
                # Stuck penalty (more lenient)
                if self.prev_pos is not None:
                    movement = point2point(current_pos, self.prev_pos)
                    if movement < 1.5:  # Very small movement
                        self.stuck_counter += 1
                        if self.stuck_counter > self.max_stuck_steps:
                            agent_reward[agent_idx] -= 3.0  # Reduced penalty
                    else:
                        self.stuck_counter = max(0, self.stuck_counter - 1)  # Gradually reduce counter
                
                # Timeout penalty (less harsh)
                if self.step_cnt >= self.max_step:
                    # Partial reward based on progress made
                    progress_ratio = (current_pos[0] - 75) / (650 - 75)
                    agent_reward[agent_idx] = -20.0 + (progress_ratio * 15.0)
            
            self.prev_pos = current_pos.copy()
        
        return agent_reward

    def get_checkpoint_reward(self, pos):
        """Reward system based on reaching key waypoints"""
        reward = 0
        
        for i, checkpoint in enumerate(self.checkpoints):
            if not self.checkpoint_reached[i]:
                dist_to_checkpoint = point2point(pos, checkpoint)
                if dist_to_checkpoint < 40:  # Close enough to checkpoint
                    self.checkpoint_reached[i] = True
                    self.current_checkpoint = i + 1
                    # Progressive rewards - later checkpoints worth more
                    reward += 10.0 * (i + 1)
                    break
        
        return reward

    def get_exploration_reward(self, pos):
        """Reward for exploring new areas"""
        if len(self.recent_positions) == 0:
            self.recent_positions.append(pos.copy())
            return 0.5
        
        # Check if this is a significantly new position
        min_distance = min([point2point(pos, old_pos) for old_pos in self.recent_positions])
        
        if min_distance > 30:  # New area
            self.recent_positions.append(pos.copy())
            return 0.5
        
        return 0

    def get_min_obstacle_distance(self, pos):
        """Get minimum distance to any green obstacle"""
        min_dist = float('inf')
        for obstacle in self.penalty:
            # Approximate obstacle position from its init_pos
            obs_center = [(obstacle.init_pos[0][0] + obstacle.init_pos[1][0]) / 2,
                         (obstacle.init_pos[0][1] + obstacle.init_pos[1][1]) / 2]
            dist = point2point(pos, obs_center)
            min_dist = min(min_dist, dist)
        return min_dist

    def is_moving_toward_obstacle(self, current_pos, previous_pos):
        """Check if agent is moving toward the nearest obstacle"""
        current_min_dist = self.get_min_obstacle_distance(current_pos)
        prev_min_dist = self.get_min_obstacle_distance(previous_pos)
        return current_min_dist < prev_min_dist

    def check_wall_collision(self, current_pos, previous_pos):
        """Check if agent collided with wall by comparing positions"""
        movement = point2point(current_pos, previous_pos)
        # If agent barely moved but tried to move, likely hit a wall
        return movement < 1.0 and self.step_cnt > 1

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            return True
        
        for agent_idx in range(self.agent_num):
            if self.agent_list[agent_idx].finished:
                return True
        
        return False

    def cross_detect(self, new_pos):
        self.info = ''
        for agent_idx in range(self.agent_num):
            agent = self.agent_list[agent_idx]
            agent_checked = False
            
            for final in self.finals:
                if final.check_cross(self.agent_pos[agent_idx], agent.r):
                    agent.color = self.cross_color
                    agent.finished = True
                    agent.alive = False
                    agent_checked = True
                    self.info = 'finished'
            
            if not agent_checked:
                for pen in self.penalty:
                    if pen.check_cross(self.agent_pos[agent_idx], agent.r):
                        agent.color = self.penalty_color
                        agent.finished = True
                        agent.alive = False
                        self.info = 'penalty'

    def get_episode_data(self):
        """Return current episode path and actions for recording"""
        return {
            'path': self.current_path.copy(),
            'actions': self.episode_actions.copy(),
            'checkpoints_reached': sum(self.checkpoint_reached),
            'final_x_position': self.agent_pos[0][0] if len(self.agent_pos) > 0 else 75,
            'success': self.info == 'finished',
            'steps': self.step_cnt
        }

    def reset(self):
        """Reset environment with slight randomization"""
        # Reset path tracking
        self.current_path = []
        self.episode_actions = []
        
        # Add small random variation to starting position to encourage exploration
        start_x = 75 + random.uniform(-5, 5)
        start_y = 300 + random.uniform(-30, 30)
        
        # Make sure starting position is valid
        start_x = max(65, min(85, start_x))
        start_y = max(270, min(430, start_y))
        
        self.map['agents'][0].position = [start_x, start_y]
        
        # Reset tracking variables
        self.prev_pos = None
        self.stuck_counter = 0
        self.obstacle_collision_count = 0
        self.stagnation_counter = 0
        
        # Reset checkpoint tracking
        self.current_checkpoint = 0
        self.checkpoint_reached = [False] * len(self.checkpoints)
        
        # Keep some memory of best progress to encourage consistent improvement
        # Don't reset best_x_progress every episode - let it decay slowly
        if hasattr(self, 'best_x_progress'):
            self.best_x_progress = max(75, self.best_x_progress * 0.95)
        else:
            self.best_x_progress = 75
        
        # Initialize recent_positions if it doesn't exist yet
        if not hasattr(self, 'recent_positions'):
            self.recent_positions = deque(maxlen=20)
        else:
            # Keep some recent positions across episodes for better exploration
            if len(self.recent_positions) > 10:
                # Keep only the most recent positions
                new_positions = deque(maxlen=20)
                for pos in list(self.recent_positions)[-5:]:
                    new_positions.append(pos)
                self.recent_positions = new_positions
        
        return super().reset()

    def render(self, info=None):
        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode=True
        
        self.viewer.draw_background()
        for w in self.map['objects']:
            self.viewer.draw_map(w)
        
        self.viewer.draw_ball(self.agent_pos, self.agent_list)
        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)
        self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        
        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)
            self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=500, upmost_y=5)
        
        debug('Step: ' + str(self.step_cnt), x=30)
        debug('Stuck Counter: ' + str(self.stuck_counter), x=30, y=50)
        debug('Current Checkpoint: ' + str(self.current_checkpoint), x=30, y=70)
        debug('Best X Progress: ' + str(int(self.best_x_progress)), x=30, y=90)
        debug('Stagnation: ' + str(self.stagnation_counter), x=30, y=110)
        if info is not None:
            debug(info, x=100)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()


# Expanded action space to match original (36 actions)
actions_map = {
        0: [-100, -30],  1: [-100, -18],  2: [-100, -6],   3: [-100, 6],   4: [-100, 18],  5: [-100, 30],
        6: [-40, -30],   7: [-40, -18],   8: [-40, -6],    9: [-40, 6],    10: [-40, 18],  11: [-40, 30],
        12: [20, -30],   13: [20, -18],   14: [20, -6],    15: [20, 6],    16: [20, 18],   17: [20, 30],
        18: [80, -30],   19: [80, -18],   20: [80, -6],    21: [80, 6],    22: [80, 18],   23: [80, 30],
        24: [140, -30],  25: [140, -18],  26: [140, -6],   27: [140, 6],   28: [140, 18],  29: [140, 30],
        30: [200, -30],  31: [200, -18],  32: [200, -6],   33: [200, 6],   34: [200, 18],  35: [200, 30]
    }

class BestPathTracker:
    """Class to track and save the best paths during training"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_reward = float('-inf')
        self.best_successful_reward = float('-inf')
        self.best_path_data = None
        self.best_successful_path_data = None
        self.all_successful_paths = []
        
        # Create directories
        os.makedirs(os.path.join(save_dir, 'best_paths'), exist_ok=True)
        
        # Initialize tracking file
        self.tracking_file = os.path.join(save_dir, 'best_paths', 'path_tracking.json')
        self.load_existing_records()
    
    def load_existing_records(self):
        """Load existing records if they exist"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                    self.best_reward = data.get('best_reward', float('-inf'))
                    self.best_successful_reward = data.get('best_successful_reward', float('-inf'))
                    print(f"Loaded existing best reward: {self.best_reward}")
                    print(f"Loaded existing best successful reward: {self.best_successful_reward}")
            except Exception as e:
                print(f"Error loading existing records: {e}")
    
    def update(self, episode, reward, episode_data):
        """Update best path records"""
        updated = False
        
        # Check if this is a new best overall reward
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_path_data = {
                'episode': episode,
                'reward': reward,
                'data': episode_data
            }
            
            # Save best path
            best_path_file = os.path.join(self.save_dir, 'best_paths', 'best_overall_path.pkl')
            with open(best_path_file, 'wb') as f:
                pickle.dump(self.best_path_data, f)
            
            print(f"NEW BEST OVERALL REWARD: {reward:.2f} at episode {episode}")
            updated = True
        
        # Check if this is a successful episode and if it's the best successful one
        if episode_data['success']:
            self.all_successful_paths.append({
                'episode': episode,
                'reward': reward,
                'data': episode_data
            })
            
            if reward > self.best_successful_reward:
                self.best_successful_reward = reward
                self.best_successful_path_data = {
                    'episode': episode,
                    'reward': reward,
                    'data': episode_data
                }
                
                # Save best successful path
                best_success_file = os.path.join(self.save_dir, 'best_paths', 'best_successful_path.pkl')
                with open(best_success_file, 'wb') as f:
                    pickle.dump(self.best_successful_path_data, f)
                
                print(f"NEW BEST SUCCESSFUL REWARD: {reward:.2f} at episode {episode}")
                updated = True
        
        # Update tracking file
        if updated:
            tracking_data = {
                'best_reward': self.best_reward,
                'best_successful_reward': self.best_successful_reward,
                'total_successful_episodes': len(self.all_successful_paths),
                'last_updated_episode': episode
            }
            
            with open(self.tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2)
        
        return updated
    
    def save_periodic_summary(self, episode):
        """Save periodic summary of all successful paths"""
        if len(self.all_successful_paths) > 0 and episode % 100 == 0:
            summary_file = os.path.join(self.save_dir, 'best_paths', f'successful_paths_summary_ep{episode}.json')
            
            summary_data = {
                'episode': episode,
                'total_successful_paths': len(self.all_successful_paths),
                'best_successful_reward': self.best_successful_reward,
                'average_successful_reward': sum(p['reward'] for p in self.all_successful_paths) / len(self.all_successful_paths),
                'successful_episodes': [p['episode'] for p in self.all_successful_paths],
                'successful_rewards': [p['reward'] for p in self.all_successful_paths]
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'best_overall_reward': self.best_reward,
            'best_successful_reward': self.best_successful_reward,
            'total_successful_episodes': len(self.all_successful_paths),
            'success_rate_recent': len([p for p in self.all_successful_paths[-100:]]) / min(100, len(self.all_successful_paths)) if self.all_successful_paths else 0
        }

parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="Learn2Avoid", type=str)
parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
parser.add_argument('--max_episodes', default=1000, type=int)  # Increased
parser.add_argument('--episode_length', default=500, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument("--save_interval", default=100, type=int)
parser.add_argument("--model_episode", default=0, type=int)
parser.add_argument("--load_model", action='store_true')
parser.add_argument("--load_run", default=2, type=int)
parser.add_argument("--load_episode", default=900, type=int)
parser.add_argument("--early_stop", action='store_true', help="Enable early stopping when convergence is detected")

device = 'cpu'
RENDER = True

def check_convergence(episode, record_win, record_reward, min_episodes=200, win_rate_threshold=0.95, reward_std_threshold=10.0, improvement_threshold=5.0):
    """
    Check for convergence based on stable, high win rates and rewards.
    """
    # 1. Check if we have enough data to make a decision
    if episode < min_episodes or len(record_win) < 100:
        return False

    # 2. Check for high and stable success rate
    current_win_rate = sum(record_win) / len(record_win)
    if current_win_rate < win_rate_threshold:
        return False

    # 3. Check for stable rewards (low standard deviation)
    reward_array = np.array(list(record_reward))
    current_reward_std = np.std(reward_array)
    if current_reward_std > reward_std_threshold:
        return False

    # 4. Check for reward plateau (no significant improvement)
    # Compare the average of the most recent 50 episodes to the 50 before that
    first_half_avg = np.mean(reward_array[:50])
    second_half_avg = np.mean(reward_array[50:])
    improvement = second_half_avg - first_half_avg
    if improvement > improvement_threshold:
        # Still improving significantly
        return False

    # If all checks pass, we have likely converged
    print("\n" + "="*50)
    print(f"CONVERGENCE DETECTED at episode {episode}!")
    print(f"  - Win Rate (last 100 ep): {current_win_rate:.3f} >= {win_rate_threshold}")
    print(f"  - Reward Std Dev (last 100 ep): {current_reward_std:.2f} <= {reward_std_threshold}")
    print(f"  - Reward Improvement (last 50 vs 50 before): {improvement:.2f} <= {improvement_threshold}")
    print("="*50 + "\n")
    return True

def main(args):
    num_agents = 1
    ctrl_agent_index = 0
    
    env = env_test()
    
    print(f'Playing game {args.game_name}')
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')
    print(f'Total agent number: {num_agents}')
    print(f'Agent control by the actor: {ctrl_agent_index}')
    
    obs_dim = 40*40
    print(f'observation dimension: {obs_dim}')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if not args.load_model:
        writer = SummaryWriter(os.path.join(str(log_dir), "{}_{} on subgames {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.algo, args.game_name)))
        save_config(args, log_dir)
        
        # Initialize best path tracker
        best_path_tracker = BestPathTracker(str(run_dir))
    
    record_win = deque(maxlen=100)
    record_reward = deque(maxlen=100)
    
    if args.load_model:
        model = PPO()
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir, episode=args.load_episode)
    else:
        model = PPO(run_dir)
        Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])
    
    episode = 0
    train_count = 0
    
    while episode < args.max_episodes:
        state = env.reset()
        if RENDER:
            env.render()
        
        obs_ctrl_agent = state[ctrl_agent_index].flatten()
        
        episode += 1
        step = 0
        Gt = 0
        episode_rewards = []
        
        while True:
            # Use standard PPO exploration (no epsilon parameter needed)
            action_ctrl_raw, action_prob = model.select_action(
                obs_ctrl_agent, 
                False if args.load_model else True
            )
            
            # Add bounds checking for action
            if action_ctrl_raw not in actions_map:
                print(f"Warning: Invalid action {action_ctrl_raw}, clipping to valid range")
                action_ctrl_raw = max(0, min(action_ctrl_raw, len(actions_map) - 1))
            
            action_ctrl = actions_map[action_ctrl_raw]
            action = [action_ctrl]
            
            next_state, reward, done, info = env.step(action)
            next_obs_ctrl_agent = next_state[ctrl_agent_index].flatten()
            
            step += 1
            post_reward = reward
            
            if not args.load_model:
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob, 
                                 post_reward[ctrl_agent_index], next_obs_ctrl_agent, done)
                model.store_transition(trans)
            
            obs_ctrl_agent = next_obs_ctrl_agent
            
            if RENDER:
                env.render()
            
            Gt += reward[ctrl_agent_index]
            episode_rewards.append(reward[ctrl_agent_index])
            
            if done:
                win_is = 1 if info == 'finished' else 0
                record_win.append(win_is)
                record_reward.append(Gt)
                
                # Get episode data for path tracking
                if not args.load_model:
                    episode_data = env.get_episode_data()
                    path_updated = best_path_tracker.update(episode, Gt, episode_data)
                    
                    # Save periodic summaries
                    if episode % 100 == 0:
                        best_path_tracker.save_periodic_summary(episode)
                        
                        # Log best path statistics
                        stats = best_path_tracker.get_stats()
                        writer.add_scalar('best_paths/best_overall_reward', stats['best_overall_reward'], episode)
                        writer.add_scalar('best_paths/best_successful_reward', stats['best_successful_reward'], episode)
                        writer.add_scalar('best_paths/total_successful_episodes', stats['total_successful_episodes'], episode)
                        writer.add_scalar('best_paths/success_rate_recent', stats['success_rate_recent'], episode)
                
                print(f"Episode: {episode}, Agent: {ctrl_agent_index}, Return: {Gt:.2f}, "
                      f"Trained: {train_count}, Win rate: {sum(record_win)/len(record_win):.3f}, "
                      f"Avg reward: {sum(record_reward)/len(record_reward):.2f}, Info: {info}, "
                      f"Checkpoints: {sum(env.checkpoint_reached)}/{len(env.checkpoints)}")
                
                if not args.load_model:
                    if args.algo == 'ppo' and len(model.buffer) >= model.batch_size:
                        model.update(episode)
                        train_count += 1
                    
                    writer.add_scalar('training Gt', Gt, episode)
                    writer.add_scalar('win_rate', sum(record_win)/len(record_win), episode)
                    writer.add_scalar('episode_length', step, episode)
                    writer.add_scalar('checkpoints_reached', sum(env.checkpoint_reached), episode)
                    writer.add_scalar('best_x_progress', env.best_x_progress, episode)
                
                # Check for convergence and stop early if needed
                if args.early_stop and not args.load_model:
                    if check_convergence(episode, record_win, record_reward):
                        print("Stopping training due to convergence.")
                        # Set episode to max_episodes to break outer loop gracefully
                        episode = args.max_episodes 
                
                break
        
        if episode % args.save_interval == 0 and not args.load_model:
            model.save(run_dir, episode)
            
            # Also save the model if we achieved a new best reward
            if not args.load_model and 'best_path_tracker' in locals():
                stats = best_path_tracker.get_stats()
                if Gt == stats['best_overall_reward']:
                    # Save special copy of the best model
                    best_model_dir = os.path.join(str(run_dir), 'best_models')
                    os.makedirs(best_model_dir, exist_ok=True)
                    model.save(best_model_dir, f'best_overall_ep{episode}')
                    print(f"Saved best overall model at episode {episode}")
                
                if info == 'finished' and Gt == stats['best_successful_reward']:
                    # Save special copy of the best successful model
                    best_model_dir = os.path.join(str(run_dir), 'best_models')
                    os.makedirs(best_model_dir, exist_ok=True)
                    model.save(best_model_dir, f'best_successful_ep{episode}')
                    print(f"Saved best successful model at episode {episode}")
    
    # Final summary
    if not args.load_model:
        print("\n" + "="*50)
        print("TRAINING COMPLETED - FINAL SUMMARY")
        print("="*50)
        
        # In case loop was broken early, set episode to final count
        episode = min(episode, args.max_episodes)
        final_stats = best_path_tracker.get_stats()
        print(f"Best overall reward achieved: {final_stats['best_overall_reward']:.2f}")
        print(f"Best successful reward achieved: {final_stats['best_successful_reward']:.2f}")
        print(f"Total successful episodes: {final_stats['total_successful_episodes']}")
        print(f"Recent success rate: {final_stats['success_rate_recent']:.3f}")
        
        # Save final summary
        final_summary = {
            'training_completed': True,
            'total_episodes': episode,
            'final_stats': final_stats,
            'final_win_rate': sum(record_win)/len(record_win) if record_win else 0,
            'final_avg_reward': sum(record_reward)/len(record_reward) if record_reward else 0
        }
        
        final_summary_file = os.path.join(str(run_dir), 'best_paths', 'final_training_summary.json')
        with open(final_summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        print(f"Final summary saved to: {final_summary_file}")
        print("="*50)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)