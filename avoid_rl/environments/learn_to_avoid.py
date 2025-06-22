# --- START OF FILE avoid_rl/environment.py ---

import math
import random
import sys
import os
from pathlib import Path
from collections import deque

# --- Sys Path Setup ---
# This setup assumes the 'avoid_rl' folder is at the same level as 'olympics_engine'
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
engine_path = os.path.join(base_dir, "olympics_engine")
sys.path.append(engine_path)
# --- End Sys Path Setup ---

from olympics_engine.core import OlympicsBase
from olympics_engine.objects import *
from olympics_engine.viewer import Viewer, debug
import pygame


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

class LearnToAvoidEnv(OlympicsBase):
    def __init__(self, map=gamemap, point_left_prob = 1):
        self.map = map
        self.gamma = 1
        self.wall_restitution = 0.5
        self.print_log = False
        self.tau = 0.1
        self.max_step = 500
        
        self.draw_obs = True
        self.show_traj = False
        
        self.cross_color = 'red'
        self.penalty_color = 'green'
        
        self.initial_pos = [75, 300]
        self.goal_pos = [650, 350]
        
        self.prev_pos = None
        self.stuck_counter = 0
        self.max_stuck_steps = 30
        
        self.obstacle_collision_count = 0
        self.recent_positions = deque(maxlen=20)
        
        self.checkpoints = [
            [130, 325], [220, 375], [320, 300], [420, 375], [520, 300], [600, 350]
        ]
        self.current_checkpoint = 0
        self.checkpoint_reached = [False] * len(self.checkpoints)
        
        self.best_x_progress = 75
        self.stagnation_counter = 0
        
        self.current_path = []
        self.episode_actions = []
        
        super(LearnToAvoidEnv, self).__init__(map)
        
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
            
            if self.agent_list[agent_idx].finished:
                if self.agent_list[agent_idx].color == self.cross_color:
                    agent_reward[agent_idx] = 500.0  # Increased success reward
                elif self.agent_list[agent_idx].color == self.penalty_color:
                    agent_reward[agent_idx] = -50.0  # Harsher obstacle penalty
                    self.obstacle_collision_count += 1
            else:
                # 1. Reduced time penalty to encourage exploration
                agent_reward[agent_idx] -= 0.001
                
                # 2. Improved distance-based reward with smoother scaling
                current_dist = point2point(current_pos, self.goal_pos)
                prev_dist = point2point(previous_pos[agent_idx], self.goal_pos)
                
                # Progress reward with exponential scaling for being closer to goal
                if current_dist < prev_dist:
                    progress = (prev_dist - current_dist)
                    # Scale reward based on how close to goal (more reward when closer)
                    distance_scale = 1.0 + (650 - current_pos[0]) / 650  # Higher reward closer to goal
                    agent_reward[agent_idx] += progress * 2.0 * distance_scale
                elif current_dist > prev_dist:
                    # Small penalty for moving away from goal
                    agent_reward[agent_idx] -= (current_dist - prev_dist) * 0.5
                
                # 3. Improved obstacle avoidance with graduated penalties
                min_obstacle_dist = self.get_min_obstacle_distance(current_pos)
                if min_obstacle_dist < 20:  # Danger zone
                    penalty = (20 - min_obstacle_dist) / 20  # Normalized penalty
                    agent_reward[agent_idx] -= penalty * 5.0
                elif min_obstacle_dist < 35:  # Warning zone
                    penalty = (35 - min_obstacle_dist) / 35
                    agent_reward[agent_idx] -= penalty * 1.0
                else:
                    # Small reward for maintaining safe distance
                    agent_reward[agent_idx] += 0.1
                
                # 4. Velocity-based reward to encourage movement
                velocity = point2point(current_pos, previous_pos[agent_idx])
                if velocity > 0.5:  # Moving
                    agent_reward[agent_idx] += min(velocity * 0.2, 1.0)  # Cap velocity reward
                elif velocity < 0.1:  # Nearly stationary
                    self.stuck_counter += 1
                    if self.stuck_counter > 10:
                        agent_reward[agent_idx] -= 0.5 * (self.stuck_counter - 10) / 10
                else:
                    self.stuck_counter = max(0, self.stuck_counter - 1)
                
                # 5. Wall avoidance with prediction
                wall_penalty = self.get_wall_proximity_penalty(current_pos)
                agent_reward[agent_idx] -= wall_penalty
                
                # 6. Improved checkpoint system with path guidance
                checkpoint_reward = self.get_improved_checkpoint_reward(current_pos)
                agent_reward[agent_idx] += checkpoint_reward
                
                # 7. Exploration bonus based on area coverage
                exploration_reward = self.get_exploration_reward_improved(current_pos)
                agent_reward[agent_idx] += exploration_reward
                
            self.prev_pos = current_pos.copy()
        
        return agent_reward

    def get_wall_proximity_penalty(self, pos):
        """Penalty for being too close to walls"""
        penalty = 0
        
        # Left wall
        if pos[0] < 70:
            penalty += (70 - pos[0]) * 0.1
        
        # Top/bottom walls
        if pos[1] < 270:
            penalty += (270 - pos[1]) * 0.1
        elif pos[1] > 430:
            penalty += (pos[1] - 430) * 0.1
        
        return penalty

    def get_improved_checkpoint_reward(self, pos):
        """Improved checkpoint system with directional guidance"""
        reward = 0
        
        # Check if approaching next checkpoint
        if self.current_checkpoint < len(self.checkpoints):
            next_checkpoint = self.checkpoints[self.current_checkpoint]
            distance = point2point(pos, next_checkpoint)
            
            # Large reward for reaching checkpoint
            if distance < 40:
                if not self.checkpoint_reached[self.current_checkpoint]:
                    self.checkpoint_reached[self.current_checkpoint] = True
                    reward += 50.0 * (self.current_checkpoint + 1)  # Progressive rewards
                    self.current_checkpoint = min(self.current_checkpoint + 1, len(self.checkpoints) - 1)
            
            # Guidance reward for moving toward next checkpoint
            elif distance < 80:
                # Reward based on how close to checkpoint
                proximity_reward = (80 - distance) / 80 * 2.0
                reward += proximity_reward
        
        return reward

    def get_exploration_reward_improved(self, pos):
        """Improved exploration reward with spatial grid"""
        if not hasattr(self, 'visited_grid'):
            # Create a grid to track visited areas
            self.visited_grid = set()
            self.grid_size = 25  # 25x25 pixel grid cells
        
        # Convert position to grid coordinates
        grid_x = int(pos[0] // self.grid_size)
        grid_y = int(pos[1] // self.grid_size)
        grid_cell = (grid_x, grid_y)
        
        # Reward for visiting new areas
        if grid_cell not in self.visited_grid:
            self.visited_grid.add(grid_cell)
            return 2.0  # Exploration bonus
        
        return 0

    # Additional method to add to the class
    def get_directional_guidance(self, current_pos, previous_pos):
        """Provide reward for moving in generally correct direction"""
        # Vector toward goal
        to_goal = [self.goal_pos[0] - current_pos[0], self.goal_pos[1] - current_pos[1]]
        
        # Movement vector
        movement = [current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1]]
        
        # Normalize vectors
        movement_mag = math.sqrt(movement[0]**2 + movement[1]**2)
        goal_mag = math.sqrt(to_goal[0]**2 + to_goal[1]**2)
        
        if movement_mag > 0 and goal_mag > 0:
            # Dot product gives us alignment
            dot_product = (movement[0] * to_goal[0] + movement[1] * to_goal[1]) / (movement_mag * goal_mag)
            
            # Reward alignment with goal direction
            if dot_product > 0:
                return dot_product * 0.5
            else:
                return dot_product * 0.2  # Smaller penalty for wrong direction
        
        return 0
    def get_checkpoint_reward(self, pos):
        """Improved checkpoint system"""
        reward = 0
        for i, checkpoint in enumerate(self.checkpoints):
            if not self.checkpoint_reached[i]:
                distance = point2point(pos, checkpoint)
                if distance < 50:  # Larger checkpoint radius
                    self.checkpoint_reached[i] = True
                    self.current_checkpoint = i + 1
                    # Increasing rewards for later checkpoints
                    reward += 15.0 * (i + 1)
                    break
                elif distance < 80:  # Approaching checkpoint
                    reward += 1.0 / (distance / 10)  # Gradual approach reward
        return reward

    def get_exploration_reward(self, pos):
        if len(self.recent_positions) == 0:
            self.recent_positions.append(pos.copy())
            return 0.5
        min_distance = min([point2point(pos, old_pos) for old_pos in self.recent_positions])
        if min_distance > 30:
            self.recent_positions.append(pos.copy())
            return 0.5
        return 0

    def get_min_obstacle_distance(self, pos):
        min_dist = float('inf')
        for obstacle in self.penalty:
            obs_center = [(obstacle.init_pos[0][0] + obstacle.init_pos[1][0]) / 2,
                          (obstacle.init_pos[0][1] + obstacle.init_pos[1][1]) / 2]
            dist = point2point(pos, obs_center)
            min_dist = min(min_dist, dist)
        return min_dist

    def is_moving_toward_obstacle(self, current_pos, previous_pos):
        return self.get_min_obstacle_distance(current_pos) < self.get_min_obstacle_distance(previous_pos)

    def check_wall_collision(self, current_pos, previous_pos):
        return point2point(current_pos, previous_pos) < 1.0 and self.step_cnt > 1
    
    def get_navigation_reward(self, current_pos, previous_pos):
        """Reward for smart navigation around obstacles"""
        reward = 0
        
        # Find nearest obstacle
        min_dist = float('inf')
        nearest_obstacle = None
        for obstacle in self.penalty:
            obs_center = [(obstacle.init_pos[0][0] + obstacle.init_pos[1][0]) / 2,
                        (obstacle.init_pos[0][1] + obstacle.init_pos[1][1]) / 2]
            dist = point2point(current_pos, obs_center)
            if dist < min_dist:
                min_dist = dist
                nearest_obstacle = obs_center
        
        if nearest_obstacle and min_dist < 50:  # If close to an obstacle
            # Calculate if agent is moving around obstacle (not directly toward/away)
            to_obstacle = [nearest_obstacle[0] - current_pos[0], nearest_obstacle[1] - current_pos[1]]
            movement = [current_pos[0] - previous_pos[0], current_pos[1] - previous_pos[1]]
            
            # Normalize vectors
            if sum(abs(x) for x in movement) > 0:
                movement_mag = math.sqrt(movement[0]**2 + movement[1]**2)
                if movement_mag > 0:
                    movement_norm = [movement[0]/movement_mag, movement[1]/movement_mag]
                    to_obstacle_mag = math.sqrt(to_obstacle[0]**2 + to_obstacle[1]**2)
                    to_obstacle_norm = [to_obstacle[0]/to_obstacle_mag, to_obstacle[1]/to_obstacle_mag]
                    
                    # Dot product - if close to 0, agent is moving perpendicular (good for navigation)
                    dot_product = movement_norm[0] * to_obstacle_norm[0] + movement_norm[1] * to_obstacle_norm[1]
                    
                    # Reward perpendicular movement when close to obstacles
                    if abs(dot_product) < 0.5:  # Moving roughly perpendicular
                        reward += 0.3
                    
                    # Small reward for moving away from obstacles
                    if dot_product < -0.3:  # Moving away
                        reward += 0.1
        
        return reward
    
    def check_wall_collision_improved(self, current_pos, previous_pos):
        """Improved wall collision detection"""
        # Check if agent is at the boundaries
        if (current_pos[0] <= 55 or current_pos[0] >= 645 or 
            current_pos[1] <= 255 or current_pos[1] >= 445):
            # Check if movement is very small (stuck against wall)
            movement = point2point(current_pos, previous_pos)
            return movement < 1.0 and self.step_cnt > 5
        return False

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
        return {
            'path': self.current_path.copy(),
            'actions': self.episode_actions.copy(),
            'checkpoints_reached': sum(self.checkpoint_reached),
            'final_x_position': self.agent_pos[0][0] if len(self.agent_pos) > 0 else 75,
            'success': self.info == 'finished',
            'steps': self.step_cnt
        }

    def reset(self):
        self.current_path = []
        self.episode_actions = []
        
        start_x = 75 + random.uniform(-5, 5)
        start_y = 300 + random.uniform(-30, 30)
        start_x = max(65, min(85, start_x))
        start_y = max(270, min(430, start_y))
        
        self.map['agents'][0].position = [start_x, start_y]
        
        self.prev_pos = None
        self.stuck_counter = 0
        self.obstacle_collision_count = 0
        self.stagnation_counter = 0
        
        self.current_checkpoint = 0
        self.checkpoint_reached = [False] * len(self.checkpoints)
        
        if hasattr(self, 'best_x_progress'):
            self.best_x_progress = max(75, self.best_x_progress * 0.95)
        else:
            self.best_x_progress = 75
        
        if not hasattr(self, 'recent_positions'):
            self.recent_positions = deque(maxlen=20)
        elif len(self.recent_positions) > 10:
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