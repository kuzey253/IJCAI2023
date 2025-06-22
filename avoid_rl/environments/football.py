# --- ANTI-REWARD-HACKING VERSION OF environments/football.py ---

import math
import numpy as np
from olympics_engine.generator import create_scenario
from scenario.football import football as FootballGame

class FootballEnv:
    def __init__(self, map=None):
        if map is None:
            Gamemap = create_scenario("football")
            self.game = FootballGame(Gamemap)
        else:
            self.game = FootballGame(map)
        
        self.agent_num = self.game.agent_num
        self.max_step = 1000
        
        # Field dimensions
        self.field_width = 720.0
        self.field_height = 480.0
        
        # Anti-reward-hacking tracking
        self.reset_tracking()

    def reset_tracking(self):
        """Reset all tracking variables"""
        self.prev_ball_pos = None
        self.prev_agent_pos = [None, None]
        self.ball_goal_progress = 0.0  # How close ball is to our goal (0-1)
        self.steps_without_progress = 0
        self.last_significant_progress = 0
        
    def reset(self):
        self.step_cnt = 0
        self.reset_tracking()
        
        obs_list_of_dicts = self.game.reset()
        self._update_tracking()
        
        return [item['agent_obs'] for item in obs_list_of_dicts]

    def step(self, action_list):
        obs_list_of_dicts, original_reward, done, info_from_game = self.game.step(action_list)
        
        if not isinstance(info_from_game, dict):
            info = {}
        else:
            info = info_from_game

        obs = [item['agent_obs'] for item in obs_list_of_dicts]

        # Update tracking before calculating rewards
        self._update_tracking()
        
        # Use the new anti-hacking reward system
        reward = self.get_anti_hacking_reward(original_reward)
        
        self.step_cnt += 1
        if self.step_cnt >= self.max_step:
            done = True
        
        if done:
            if original_reward[0] > original_reward[1]:
                info['win_signal'] = 1
                info['game_result'] = 'agent_0_win'
            elif original_reward[1] > original_reward[0]:
                info['win_signal'] = -1
                info['game_result'] = 'agent_1_win'
            else:
                info['win_signal'] = 0
                info['game_result'] = 'draw'
            
            info['total_steps'] = self.step_cnt
            info['final_ball_progress'] = self.ball_goal_progress

        return obs, reward, done, info

    def _update_tracking(self):
        """Update tracking variables for anti-hacking measures"""
        ball_pos = self._get_ball_position()
        if ball_pos is None:
            return
            
        # Calculate ball progress toward Agent 0's target goal (right side)
        # 0.0 = at Agent 0's goal, 1.0 = at Agent 1's goal (Agent 0's target)
        new_progress = ball_pos[0] / self.field_width
        
        if self.ball_goal_progress is not None:
            progress_change = new_progress - self.ball_goal_progress
            
            # Check for meaningful progress (more than 1% of field)
            if abs(progress_change) > 0.01:
                self.last_significant_progress = self.step_cnt
                self.steps_without_progress = 0
            else:
                self.steps_without_progress += 1
        
        self.ball_goal_progress = new_progress
        
        # Store previous positions
        if hasattr(self.game, 'agent_pos'):
            self.prev_agent_pos = [pos.copy() if pos is not None else None 
                                 for pos in self.game.agent_pos[:2]]
        self.prev_ball_pos = ball_pos.copy()

    def _get_ball_position(self):
        """Get the current ball position"""
        if hasattr(self.game, 'agent_pos') and hasattr(self.game, 'agent_list'):
            for i, agent in enumerate(self.game.agent_list):
                if agent.type == 'ball':
                    return np.array(self.game.agent_pos[i])
        return None

    def get_anti_hacking_reward(self, original_reward):
        """Anti-reward-hacking reward system"""
        agent_reward = [0.0, 0.0]
        
        # DOMINANT goal rewards - these should dwarf all other rewards
        if original_reward[0] > original_reward[1]:  # Agent 0 scores
            agent_reward[0] = 1000.0  # Massive positive reward
            agent_reward[1] = -1000.0
            return agent_reward  # Return immediately, no other rewards needed
            
        elif original_reward[1] > original_reward[0]:  # Agent 1 scores  
            agent_reward[0] = -1000.0  # Massive penalty
            agent_reward[1] = 1000.0
            return agent_reward  # Return immediately
        
        # If no goal, use sparse reward system focused on meaningful progress
        agent_reward[0] = self._calculate_sparse_reward(0)
        agent_reward[1] = self._calculate_sparse_reward(1)
        
        return agent_reward

    def _calculate_sparse_reward(self, agent_idx):
        """Sparse reward system that only rewards meaningful progress"""
        reward = 0.0
        
        # Only Agent 0 gets progress rewards (Agent 1 is opponent)
        if agent_idx != 0:
            return reward
            
        ball_pos = self._get_ball_position()
        if ball_pos is None or not hasattr(self.game, 'agent_pos'):
            return reward
            
        agent_pos = np.array(self.game.agent_pos[agent_idx])
        
        # 1. SIGNIFICANT ball progress reward (only for major advances)
        if self.prev_ball_pos is not None:
            prev_progress = self.prev_ball_pos[0] / self.field_width
            curr_progress = ball_pos[0] / self.field_width
            progress_change = curr_progress - prev_progress
            
            # Only reward significant progress toward goal (> 2% of field)
            if progress_change > 0.02:
                reward += progress_change * 10.0  # Max ~0.2 per step
        
        # 2. Goal area proximity (only when very close)
        goal_distance = abs(ball_pos[0] - self.field_width)
        if goal_distance < 100.0:  # Within 100 units of goal
            proximity_bonus = (100.0 - goal_distance) / 100.0 * 0.5
            reward += proximity_bonus
        
        # 3. Anti-stagnation penalty
        if self.steps_without_progress > 50:  # No progress for 50 steps
            stagnation_penalty = (self.steps_without_progress - 50) * 0.01
            reward -= min(stagnation_penalty, 2.0)  # Cap penalty at -2.0
        
        # 4. Timeout approaching penalty (encourage urgency)
        if self.step_cnt > self.max_step * 0.8:  # Last 20% of episode
            time_pressure = (self.step_cnt - self.max_step * 0.8) / (self.max_step * 0.2)
            reward -= time_pressure * 0.1
        
        return reward

    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)

    def close(self):
        if hasattr(self.game, 'close') and callable(self.game.close):
            self.game.close()

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]

# Alternative: Even more extreme sparse reward version
class UltraSparseFootballEnv(FootballEnv):
    """Ultra-sparse reward version - only goals and major milestones matter"""
    
    def get_anti_hacking_reward(self, original_reward):
        """Ultra-sparse: only goals and crossing half-field matter"""
        agent_reward = [0.0, 0.0]
        
        # Goals are the only major reward
        if original_reward[0] > original_reward[1]:
            agent_reward[0] = 1000.0
            agent_reward[1] = -1000.0
            return agent_reward
            
        elif original_reward[1] > original_reward[0]:
            agent_reward[0] = -1000.0
            agent_reward[1] = 1000.0
            return agent_reward
        
        # Only reward crossing major field milestones
        ball_pos = self._get_ball_position()
        if ball_pos is not None:
            ball_x = ball_pos[0]
            
            # Reward for getting ball past half-field (once per episode)
            if ball_x > self.field_width * 0.5 and not hasattr(self, '_crossed_half'):
                agent_reward[0] += 10.0
                self._crossed_half = True
            
            # Reward for getting ball to final quarter (once per episode)
            if ball_x > self.field_width * 0.75 and not hasattr(self, '_reached_quarter'):
                agent_reward[0] += 20.0
                self._reached_quarter = True
        
        return agent_reward
    
    def reset(self):
        # Reset milestone tracking
        if hasattr(self, '_crossed_half'):
            delattr(self, '_crossed_half')
        if hasattr(self, '_reached_quarter'):
            delattr(self, '_reached_quarter')
        return super().reset()