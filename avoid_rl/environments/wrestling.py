# --- IMPROVED Wrestling Environment ---

import math
import numpy as np
from olympics_engine.generator import create_scenario
from scenario.wrestling import wrestling as WrestlingGame

class WrestlingEnv:
    def __init__(self, map=None):
        if map is None:
            Gamemap = create_scenario("wrestling")
            self.game = WrestlingGame(Gamemap)
        else:
            self.game = WrestlingGame(map)
        
        self.agent_num = self.game.agent_num
        self.max_step = 1000
        
        # Wrestling-specific parameters
        self.ring_center = np.array([300, 350])
        self.ring_radius = 100.0
        
        # Enhanced tracking variables
        self.reset_tracking()
        
        # Curriculum learning parameters
        self.training_phase = 0  # 0: basic movement, 1: approach, 2: push, 3: advanced
        self.phase_step_counts = [0, 0, 0, 0]

    def reset_tracking(self):
        """Reset tracking variables for the episode"""
        self.prev_positions = {'agent': None, 'opponent': None}
        self.prev_opponent_dist_to_center = None
        self.prev_agent_dist_to_center = None
        self.steps_without_progress = 0
        self.consecutive_approach_steps = 0
        self.engagement_time = 0
        self.max_opponent_distance_reached = 0

    def reset(self):
        self.step_cnt = 0
        self.reset_tracking()
        
        obs_list_of_dicts = self.game.reset()
        self._update_tracking()
        
        # Enhanced observation with relative positioning
        enhanced_obs = []
        for i, obs_dict in enumerate(obs_list_of_dicts):
            enhanced_obs.append(self._enhance_observation(obs_dict['agent_obs'], i))
        
        return enhanced_obs

    def _enhance_observation(self, base_obs, agent_id):
        """Add relative positioning and tactical information to observations"""
        if not hasattr(self.game, 'agent_pos') or len(self.game.agent_pos) < 2:
            return base_obs
            
        agent_pos = np.array(self.game.agent_pos[agent_id])
        opponent_pos = np.array(self.game.agent_pos[1 - agent_id])
        
        # Relative positioning features
        relative_pos = opponent_pos - agent_pos
        distance_to_opponent = np.linalg.norm(relative_pos)
        
        # Ring positioning features
        agent_dist_to_center = np.linalg.norm(agent_pos - self.ring_center)
        opponent_dist_to_center = np.linalg.norm(opponent_pos - self.ring_center)
        
        # Tactical features
        angle_to_opponent = math.atan2(relative_pos[1], relative_pos[0])
        angle_to_center = math.atan2((self.ring_center - agent_pos)[1], 
                                   (self.ring_center - agent_pos)[0])
        
        # Enhanced observation vector
        enhanced_features = np.array([
            # Relative positioning (normalized)
            relative_pos[0] / 200.0,  # Normalize by typical ring size
            relative_pos[1] / 200.0,
            distance_to_opponent / 200.0,
            
            # Ring positioning (normalized)
            agent_dist_to_center / self.ring_radius,
            opponent_dist_to_center / self.ring_radius,
            
            # Tactical angles
            math.cos(angle_to_opponent),
            math.sin(angle_to_opponent),
            math.cos(angle_to_center),
            math.sin(angle_to_center),
            
            # Progress indicators
            self.consecutive_approach_steps / 100.0,  # Normalize
            self.engagement_time / 1000.0,  # Normalize
        ])
        
        # Handle different observation formats
        if isinstance(base_obs, np.ndarray):
            base_flat = base_obs.flatten()  # Flatten in case it's multi-dimensional
        else:
            base_flat = np.array(base_obs).flatten()
            
        # Concatenate flattened base observation with enhanced features
        return np.concatenate([base_flat, enhanced_features])

    def step(self, action_list):
        obs_list_of_dicts, original_reward, done, info_from_game = self.game.step(action_list)
        
        if not isinstance(info_from_game, dict):
            info = {}
        else:
            info = info_from_game

        # Enhanced observations
        obs = []
        for i, obs_dict in enumerate(obs_list_of_dicts):
            obs.append(self._enhance_observation(obs_dict['agent_obs'], i))

        self._update_tracking()
        
        # Curriculum-based reward shaping
        reward = self.get_curriculum_shaped_reward(original_reward)
        
        self.step_cnt += 1
        if self.step_cnt >= self.max_step:
            done = True
        
        if done:
            info['win_signal'] = original_reward[1]
            info['episode_stats'] = self._get_episode_stats()
            # Update curriculum based on performance
            self._update_curriculum(info)

        return obs, reward, done, info

    def _update_tracking(self):
        """Enhanced tracking with more detailed state information"""
        if not hasattr(self.game, 'agent_pos') or len(self.game.agent_pos) < 2:
            return
            
        agent_pos = np.array(self.game.agent_pos[0])
        opponent_pos = np.array(self.game.agent_pos[1])
        
        # Distance tracking
        new_agent_dist = np.linalg.norm(agent_pos - self.ring_center)
        new_opponent_dist = np.linalg.norm(opponent_pos - self.ring_center)
        
        # Check for progress in pushing opponent out
        if self.prev_opponent_dist_to_center is not None:
            if new_opponent_dist > self.prev_opponent_dist_to_center + 1.0:
                self.steps_without_progress = 0
            else:
                self.steps_without_progress += 1
        
        # Track approach behavior
        distance_to_opponent = np.linalg.norm(opponent_pos - agent_pos)
        if distance_to_opponent < 50:  # Close engagement
            self.engagement_time += 1
            if (self.prev_positions['agent'] is not None and 
                np.linalg.norm(agent_pos - opponent_pos) < 
                np.linalg.norm(self.prev_positions['agent'] - self.prev_positions['opponent'])):
                self.consecutive_approach_steps += 1
            else:
                self.consecutive_approach_steps = max(0, self.consecutive_approach_steps - 1)
        else:
            self.consecutive_approach_steps = max(0, self.consecutive_approach_steps - 1)
        
        # Track maximum distance opponent has been pushed
        self.max_opponent_distance_reached = max(self.max_opponent_distance_reached, new_opponent_dist)
        
        # Update previous positions
        self.prev_positions['agent'] = agent_pos.copy()
        self.prev_positions['opponent'] = opponent_pos.copy()
        self.prev_agent_dist_to_center = new_agent_dist
        self.prev_opponent_dist_to_center = new_opponent_dist

    def get_curriculum_shaped_reward(self, original_reward):
        """Multi-phase curriculum learning reward system, tuned for a DEFENSIVE style."""
        agent_reward = [0.0, 0.0]
        win_signal = original_reward[1]

        # --- 1. Terminal rewards are always dominant ---
        if win_signal == 1:
            agent_reward[0] = 100.0
            return agent_reward
        elif win_signal == -1:
            agent_reward[0] = -100.0
            return agent_reward

        # Early exit if game state is not available
        if not hasattr(self.game, 'agent_pos') or len(self.game.agent_pos) < 2:
            return agent_reward

        # --- 2. Extract state variables ---
        agent_pos = np.array(self.game.agent_pos[0])
        opponent_pos = np.array(self.game.agent_pos[1])
        agent_vel = np.array(self.game.agent_v[0])
        
        # --- 3. Penalize High Velocity (Discourage Rushing) ---
        # This is a key part of teaching a "steady" style.
        speed = np.linalg.norm(agent_vel)
        # We want to penalize speeds above a certain "steady" threshold, e.g., 50
        if speed > 50:
            # The penalty increases quadratically with excessive speed
            speed_penalty = ((speed - 50) / 100.0)**2 * 0.1 
            agent_reward[0] -= speed_penalty

        # --- 4. Reward a Central, Defensive Position ---
        # This is the core of the "defensive and centered" style.
        agent_dist_from_center = np.linalg.norm(agent_pos - self.ring_center)
        
        # We define a "golden zone" in the center of the ring.
        golden_zone_radius = self.ring_radius * 0.3 # e.g., inner 30% of the ring
        
        if agent_dist_from_center < golden_zone_radius:
            # Strong, continuous reward for being in the center
            center_reward = (1 - (agent_dist_from_center / golden_zone_radius)) * 0.2
            agent_reward[0] += center_reward
        else:
            # Penalty for being outside the golden zone, pushing agent back to center
            edge_penalty = (agent_dist_from_center / self.ring_radius) * 0.05
            agent_reward[0] -= edge_penalty

        # --- 5. Phase-Specific Rewards (Curriculum) ---
        # This curriculum now builds on top of the defensive base style.
        
        # Phase 0 & 1: Learn to control the center
        if self.training_phase < 2:
            # The base rewards for being centered are enough for these phases.
            # No additional rewards needed.
            pass

        # Phase 2: Engagement and Counter-Pushing
        elif self.training_phase == 2:
            distance_to_opponent = np.linalg.norm(opponent_pos - agent_pos)
            if distance_to_opponent < 40: # When engaged
                agent_reward[0] += 0.1 # Small bonus for engagement
                
                # Reward pushing, but only as a defensive reaction
                if self.prev_opponent_dist_to_center is not None:
                    opponent_dist = np.linalg.norm(opponent_pos - self.ring_center)
                    progress = opponent_dist - self.prev_opponent_dist_to_center
                    if progress > 0: # If we successfully pushed them
                        agent_reward[0] += progress * 0.5 # Simple push reward
        
        # Phase 3: Advanced Tactical Play
        else:
            # Reward for maintaining a superior position (being between the opponent and the center)
            opponent_dist = np.linalg.norm(opponent_pos - self.ring_center)
            if agent_dist_from_center < opponent_dist:
                agent_reward[0] += 0.1

            # Reward pushing, but only when it's tactically sound
            if self.prev_opponent_dist_to_center is not None:
                progress = opponent_dist - self.prev_opponent_dist_to_center
                if progress > 0:
                    # Give a bigger reward for pushing when you have the positional advantage
                    tactical_bonus = 1.0 + max(0, (opponent_dist - agent_dist_from_center) / self.ring_radius)
                    agent_reward[0] += progress * tactical_bonus * 0.8

        return agent_reward

    def _get_episode_stats(self):
        """Get statistics for curriculum learning"""
        return {
            'max_opponent_distance': self.max_opponent_distance_reached,
            'engagement_time': self.engagement_time,
            'steps_without_progress': self.steps_without_progress,
            'approach_consistency': self.consecutive_approach_steps
        }

    def _update_curriculum(self, info):
        """Update curriculum phase based on performance"""
        stats = info.get('episode_stats', {})
        win_signal = info.get('win_signal', 0)
        
        # Simple phase progression logic
        phase_thresholds = {
            0: {'engagement_time': 50},  # Must engage for 50 steps
            1: {'max_opponent_distance': self.ring_radius * 0.7},  # Must push opponent 70% to edge
            2: {'win_rate': 0.3}  # Must win 30% of episodes
        }
        
        # This is a simplified example - in practice, you'd track performance over multiple episodes
        current_phase_requirements = phase_thresholds.get(self.training_phase, {})
        
        # Increment phase step count
        if self.training_phase < len(self.phase_step_counts):
            self.phase_step_counts[self.training_phase] += 1

    def set_training_phase(self, phase):
        """Manually set training phase for curriculum learning"""
        self.training_phase = max(0, min(3, phase))

    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)
        
    def close(self):
        if hasattr(self.game, 'close') and callable(self.game.close):
            self.game.close()

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]

# Additional helper class for opponent diversity
class OpponentManager:
    """Manages different opponent strategies for training diversity"""
    
    def __init__(self):
        self.strategies = ['random', 'defensive', 'aggressive', 'smart']
        self.current_strategy = 'random'
    
    def get_opponent_action(self, obs, agent_pos, opponent_pos, ring_center, ring_radius):
        """Generate opponent action based on current strategy"""
        if self.current_strategy == 'random':
            return np.random.randint(0, 4)  # Assuming 4 possible actions
        
        elif self.current_strategy == 'defensive':
            # Try to stay near center
            opponent_dist = np.linalg.norm(opponent_pos - ring_center)
            if opponent_dist > ring_radius * 0.3:
                # Move towards center
                direction = ring_center - opponent_pos
                return self._direction_to_action(direction)
            return 0  # Stay still if near center
        
        elif self.current_strategy == 'aggressive':
            # Always move towards agent
            direction = agent_pos - opponent_pos
            return self._direction_to_action(direction)
        
        elif self.current_strategy == 'smart':
            # Simple tactical behavior
            agent_dist = np.linalg.norm(agent_pos - ring_center)
            opponent_dist = np.linalg.norm(opponent_pos - ring_center)
            
            if opponent_dist > agent_dist:
                # If further from center, try to get closer to center
                direction = ring_center - opponent_pos
            else:
                # If closer to center, try to push agent out
                direction = agent_pos - opponent_pos
            
            return self._direction_to_action(direction)
        
        return 0
    
    def _direction_to_action(self, direction):
        """Convert direction vector to discrete action"""
        if np.linalg.norm(direction) < 1e-6:
            return 0
        
        angle = math.atan2(direction[1], direction[0])
        # Convert angle to discrete action (assuming 4 directions)
        angle_deg = math.degrees(angle) % 360
        
        if angle_deg < 45 or angle_deg >= 315:
            return 1  # Right
        elif angle_deg < 135:
            return 2  # Up
        elif angle_deg < 225:
            return 3  # Left
        else:
            return 0  # Down
    
    def set_strategy(self, strategy):
        """Set opponent strategy"""
        if strategy in self.strategies:
            self.current_strategy = strategy