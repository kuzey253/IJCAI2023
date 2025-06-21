# --- START OF FILE environments/wrestling.py ---

import math
from olympics_engine.generator import create_scenario
from scenario import wrestling
from olympics_engine.core import OlympicsBase

class WrestlingEnv(OlympicsBase):
    def __init__(self, map=None):
        if map is None:
            # Use default map from the scenario if none is provided
            map = wrestling.map

        self.game = wrestling.Wrestling(map)
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
        """
        Reward is shaped based on winning and a small time penalty.
        Returns a list of rewards, one for each agent.
        """
        agent_reward = [0. for _ in range(self.agent_num)]
        
        # Win/loss reward for agent 0
        if original_reward[1] > 0:  # Agent 0 wins
            agent_reward[0] = 100.0
        elif original_reward[1] < 0: # Agent 0 loses
            agent_reward[0] = -100.0
        
        # Small penalty for each step to encourage finishing quickly
        agent_reward[0] -= 0.1

        # Reward for agent 1
        if original_reward[1] < 0: # Agent 1 wins
            agent_reward[1] = 100.0
        elif original_reward[1] > 0: # Agent 1 loses
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