# --- START OF FILE environments/billiard_competition.py ---

import math
import numpy as np
from olympics_engine.generator import create_scenario
from scenario.billiard_competition import billiard_competition as BilliardGame

class BilliardCompetitionEnv:
    def __init__(self, map=None):
        if map is None:
            Gamemap = create_scenario("billiard-competition")
            self.game = BilliardGame(Gamemap)
        else:
            self.game = BilliardGame(map)
        
        self.agent_num = 2  # This game always has 2 competing agents
        self.max_step = self.game.max_step

    def reset(self):
        self.step_cnt = 0
        # The base game's reset handles all the complex setup
        obs_list_of_dicts = self.game.reset()
        
        # The obs format is already [{"agent_obs": obs, "id": "team_X"}, ...]
        return [item['agent_obs'] for item in obs_list_of_dicts]

    def step(self, action_list):
        # The base game handles turn-taking, so we pass both actions
        obs_list_of_dicts, original_reward, done, info_from_game = self.game.step(action_list)
        
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
            # Determine winner from final scores
            if self.game.total_score[0] > self.game.total_score[1]:
                info['win_signal'] = 1
            elif self.game.total_score[1] > self.game.total_score[0]:
                info['win_signal'] = -1
            else:
                info['win_signal'] = 0

        return obs, reward, done, info

    def get_shaped_reward(self, original_reward):
        """
        The base game's reward is already well-shaped for RL.
        original_reward = [reward_A0, reward_A1]
        - It includes penalties for potting the white ball (-10).
        - It includes rewards for potting colored balls (+1).
        
        We can simply pass this through or slightly scale it.
        The base game reward is already divided by 100, so let's scale it back up
        for consistency with our other environments.
        """
        agent_reward = [r * 100.0 for r in original_reward]
        
        # Add a small incentive for the controlled agent (0) to hit something.
        # The base game state needs to be inspected for this.
        # This is an advanced addition, for now, we pass the scaled rewards.

        return agent_reward
        
    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)
        
    def close(self):
        if hasattr(self.game, 'close') and callable(self.game.close):
            self.game.close()

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]