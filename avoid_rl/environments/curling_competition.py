# --- START OF FILE environments/curling_competition.py ---
import random
import math
import numpy as np
from olympics_engine.generator import create_scenario
from scenario.curling_competition import curling_competition as CurlingGame

class CurlingCompetitionEnv:
    def __init__(self, map=None):
        if map is None:
            Gamemap = create_scenario("curling-competition")
            self.game = CurlingGame(Gamemap)
        else:
            self.game = CurlingGame(map)
        
        self.agent_num = 2 # This game always has 2 competing agents
        self.max_step = self.game.max_step

    def reset(self):
        self.step_cnt = 0
        obs_list_of_dicts = self.game.reset()
        # The obs format already contains {"agent_obs": obs, "id": "team_X"}
        return [item['agent_obs'] for item in obs_list_of_dicts]

    def step(self, action_list):
        # The base game returns obs for the CURRENT player. The other is None.
        obs_list_of_dicts, original_reward, done, info_from_game = self.game.step(action_list)
        
        if not isinstance(info_from_game, dict):
            info = {}
        else:
            info = info_from_game

        # Re-format observations to be consistent for the PPO agent
        obs = [np.zeros_like(obs_list_of_dicts[1]['agent_obs']), # Placeholder for agent 0
               np.zeros_like(obs_list_of_dicts[0]['agent_obs'])] # Placeholder for agent 1
        
        if obs_list_of_dicts[0]['agent_obs'] is not None and not isinstance(obs_list_of_dicts[0]['agent_obs'], int):
            obs[0] = obs_list_of_dicts[0]['agent_obs']
        if obs_list_of_dicts[1]['agent_obs'] is not None and not isinstance(obs_list_of_dicts[1]['agent_obs'], int):
            obs[1] = obs_list_of_dicts[1]['agent_obs']

        reward = self.get_shaped_reward(original_reward)
        
        self.step_cnt += 1
        # The base game handles max steps and round termination internally.
        
        if done:
            # The base game's check_win() returns '0', '1', or '-1'
            winner = self.game.check_win()
            if winner == '0':
                info['win_signal'] = 1
            elif winner == '1':
                info['win_signal'] = -1
            else:
                info['win_signal'] = 0

        return obs, reward, done, info

    def get_shaped_reward(self, original_reward):
        """
        The base game for curling provides round-based rewards.
        original_reward = [round_score_A0, round_score_A1]
        A win is +100 at the very end.
        
        We can shape this by providing a dense reward for getting stones
        closer to the center than the opponent.
        """
        agent_reward = [r for r in original_reward] # Start with the base game's rewards
        
        # Dense shaping reward
        # This requires finding the closest stone of each color to the center
        center = np.array(self.game.center)
        min_dist_team0 = float('inf')
        min_dist_team1 = float('inf')

        for i, agent in enumerate(self.game.agent_list):
            if agent.type == 'ball': # Thrown stones become balls
                dist = np.linalg.norm(np.array(self.game.agent_pos[i]) - center)
                if agent.color == self.game.team_0_color:
                    min_dist_team0 = min(min_dist_team0, dist)
                elif agent.color == self.game.team_1_color:
                    min_dist_team1 = min(min_dist_team1, dist)
        
        # Reward for having the closest stone (the "shot rock")
        if min_dist_team0 < min_dist_team1:
            # Reward is inversely proportional to the distance
            agent_reward[0] += (1.0 / (min_dist_team0 + 1.0)) * 0.1
        elif min_dist_team1 < min_dist_team0:
            agent_reward[1] += (1.0 / (min_dist_team1 + 1.0)) * 0.1
            
        return agent_reward
        
    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)
        
    def close(self):
        if hasattr(self.game, 'close') and callable(self.game.close):
            self.game.close()

    def seed(self, seed=None):
        # The curling env doesn't seem to have a set_seed method,
        # so we seed the global random module which it uses internally.
        random.seed(seed)
        np.random.seed(seed)
        return [seed]