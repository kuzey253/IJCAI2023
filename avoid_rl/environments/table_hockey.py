# --- START OF FILE environments/table_hockey.py ---

import math
import numpy as np
from olympics_engine.generator import create_scenario
# Import the base game class correctly
from scenario.table_hockey import table_hockey as TableHockeyGame

class TableHockeyEnv:
    def __init__(self, map=None):
        if map is None:
            Gamemap = create_scenario("table-hockey")
            self.game = TableHockeyGame(Gamemap)
        else:
            self.game = TableHockeyGame(map)
        
        self.agent_num = self.game.agent_num
        self.max_step = 1000
        
        # --- NEW: Find the puck's index ---
        self.puck_idx = -1
        for i, agent in enumerate(self.game.agent_list):
            if agent.type == 'ball':
                self.puck_idx = i
                break
        if self.puck_idx == -1:
            raise ValueError("Puck not found in agent list.")
        
        # Field dimensions for normalization
        self.field_width = 700.0 
        
        self.reset_tracking()

    def reset_tracking(self):
        self.last_hit_agent_idx = None

    def reset(self):
        self.step_cnt = 0
        self.reset_tracking()
        obs_list_of_dicts = self.game.reset()
        return [item['agent_obs'] for item in obs_list_of_dicts]

    def step(self, action_list):
        # Use the puck_idx to get the puck's velocity
        prev_puck_vel_norm = np.linalg.norm(self.game.agent_v[self.puck_idx])
        
        obs_list_of_dicts, original_reward, done, info_from_game = self.game.step(action_list)
        
        # Check who hit the puck this step
        puck_vel_norm = np.linalg.norm(self.game.agent_v[self.puck_idx])
        if puck_vel_norm > prev_puck_vel_norm + 1.0:
             self.last_hit_agent_idx = self._get_closest_agent_to_puck()

        if not isinstance(info_from_game, dict):
            info = {}
        else:
            info = info_from_game

        obs = [item['agent_obs'] for item in obs_list_of_dicts]
        reward = self.get_shaped_reward(original_reward)
        
        self.step_cnt += 1
        if self.step_cnt >= self.max_step:
            done = True
        
        if done:
            info['win_signal'] = original_reward[1]

        return obs, reward, done, info

    def _get_puck_pos(self):
        return np.array(self.game.agent_pos[self.puck_idx])

    def _get_closest_agent_to_puck(self):
        puck_pos = self._get_puck_pos()
        agent0_pos = np.array(self.game.agent_pos[0])
        agent1_pos = np.array(self.game.agent_pos[1])
        return 0 if np.linalg.norm(puck_pos - agent0_pos) < np.linalg.norm(puck_pos - agent1_pos) else 1

    def get_shaped_reward(self, original_reward):
        agent_reward = [0.0, 0.0]
        win_signal = original_reward[1]

        if win_signal == 1:
            agent_reward[0] = 100.0
            agent_reward[1] = -100.0
            return agent_reward
        elif win_signal == -1:
            agent_reward[0] = -100.0
            agent_reward[1] = 100.0
            return agent_reward

        puck_pos = self._get_puck_pos()
        agent0_pos = np.array(self.game.agent_pos[0])
        agent1_pos = np.array(self.game.agent_pos[1])

        offensive_pressure_reward = ((puck_pos[0] - self.field_width / 2) / (self.field_width / 2)) * 0.1
        agent_reward[0] += offensive_pressure_reward
        agent_reward[1] -= offensive_pressure_reward

        if self._get_closest_agent_to_puck() == 0:
            agent_reward[0] += 0.05
        else:
            agent_reward[1] += 0.05
        
        if self.last_hit_agent_idx is not None:
             agent_reward[self.last_hit_agent_idx] += 0.1

        if puck_pos[0] < self.field_width / 2:
            if agent0_pos[0] < puck_pos[0]:
                agent_reward[0] += 0.05

        if puck_pos[0] > self.field_width / 2:
            if agent1_pos[0] > puck_pos[0]:
                agent_reward[1] += 0.05

        return agent_reward
        
    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)
        
    def close(self):
        if hasattr(self.game, 'close') and callable(self.game.close):
            self.game.close()

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]