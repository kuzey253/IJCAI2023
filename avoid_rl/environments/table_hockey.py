# --- START OF FILE environments/table_hockey.py ---

import math
import numpy as np
from olympics_engine.generator import create_scenario
from scenario import table_hockey
from olympics_engine.core import OlympicsBase

class TableHockeyEnv(OlympicsBase):
    def __init__(self, map=None):
        if map is None:
            # Use default map from the scenario if none is provided
            map = table_hockey.map
        
        self.game = table_hockey.TableHockey(map)
        super().__init__(self.game.meta_map)
        
        self.max_step = 1000

    def reset(self):
        self.step_cnt = 0
        return self.game.reset()

    def step(self, action_list):
        obs, original_reward, done, info = self.game.step(action_list)
        reward = self.get_reward(original_reward)
        
        self.step_cnt += 1
        if self.step_cnt >= self.max_step:
            done = True
        
        return obs, reward, done, info

    def get_reward(self, original_reward):
        agent_reward = [0. for _ in range(self.agent_num)]

        # Agent 1 (controlled agent)
        if original_reward[1] == 1:  # agent 1 wins
            agent_reward[0] = 100.0
        elif original_reward[1] == -1:  # agent 1 loses
            agent_reward[0] = -100.0
        else:
            # Reward for puck moving towards opponent's goal (positive x-direction)
            puck_pos_x = self.game.puck_pos[0]
            agent_reward[0] += (puck_pos_x / 1000.0) * 0.2 # Dense reward

            # Reward for being close to the puck
            agent_pos = np.array(self.game.agent_pos[0])
            puck_pos = np.array(self.game.puck_pos)
            dist_to_puck = np.linalg.norm(agent_pos - puck_pos)
            agent_reward[0] += (1.0 / (dist_to_puck + 1.0)) * 0.1

        # Agent 2 (opponent) gets a simple reward
        if original_reward[1] == -1:
            agent_reward[1] = 100.0
        elif original_reward[1] == 1:
            agent_reward[1] = -100.0

        return agent_reward
        
    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)

    def close(self):
        self.game.close()

    def seed(self, seed=None):
        # Pass the call to the underlying game object
        # This assumes self.game has a working .seed() method that returns a list
        self.game.set_seed(seed)
        return [seed]