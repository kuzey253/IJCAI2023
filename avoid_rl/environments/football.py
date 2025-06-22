# --- START OF FILE environments/football.py ---

import math
import numpy as np
from olympics_engine.generator import create_scenario
from scenario import football
from olympics_engine.core import OlympicsBase

class FootballEnv(OlympicsBase):
    def __init__(self, map=None):
        if map is None:
            Gamemap = create_scenario("football")
            self.game = football.Football(Gamemap)
        else:
            self.game = football.Football(map)
        
        super().__init__(self.game.meta_map)
        
        # RL specific
        self.max_step = 1000

    def reset(self):
        self.step_cnt = 0
        return self.game.reset()

    def step(self, action_list):
        obs, original_reward, done, info = self.game.step(action_list)
        
        # Custom Reward Shaping
        reward = self.get_reward(original_reward, obs)
        
        # Custom Done Condition
        self.step_cnt += 1
        if self.step_cnt >= self.max_step:
            done = True
        
        return obs, reward, done, info

    def get_reward(self, original_reward, obs):
        """
        Reward is shaped based on winning, progress, and proximity to the ball.
        original_reward is a list [total_reward, win_reward]
        """
        agent_reward = [0. for _ in range(self.agent_num)]

        # Agent 1 (controlled agent)
        if original_reward[1] == 1:  # agent 1 wins (scores a goal)
            agent_reward[0] = 100.0
        elif original_reward[1] == -1:  # agent 1 loses (concedes a goal)
            agent_reward[0] = -100.0
        else:
            # Reward for moving towards opponent's goal
            pos_x = self.game.agent_pos[0][0]
            agent_reward[0] += (pos_x / 1000.0) * 0.1

            # Reward for being close to the ball
            agent_pos = np.array(self.game.agent_pos[0])
            ball_pos = np.array(self.game.ball_pos)
            dist_to_ball = np.linalg.norm(agent_pos - ball_pos)
            agent_reward[0] += (1.0 / (dist_to_ball + 1.0)) * 0.2  # Max reward of 0.2

        # Agent 2 (opponent) gets a simple reward
        if original_reward[1] == -1: # opponent wins
            agent_reward[1] = 100.0
        elif original_reward[1] == 1: # opponent loses
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