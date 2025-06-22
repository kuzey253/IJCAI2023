# --- START OF FILE environments/running.py ---

import math
from olympics_engine.generator import create_scenario
from scenario.running import Running
# DO NOT import or inherit from OlympicsBase here. It's a wrapper.
# from olympics_engine.core import OlympicsBase <-- REMOVE THIS

class RunningEnv: # Inherit from object (default in Python 3)
    def __init__(self, map=None):
        if map is None:
            Gamemap = create_scenario("running")
            self.game = Running(Gamemap)
        else:
            self.game = Running(map)
        
        # This super() call was the problem. REMOVE IT.
        # super().__init__(self.game.meta_map)
        
        # The wrapper needs to expose the number of agents from the game object
        self.agent_num = self.game.agent_num

        # RL specific properties
        self.max_step = 1000

        # --- CHECKPOINT SETUP ---
        self.checkpoints = [250, 500, 750, 950]
        self.checkpoint_reward_value = 25.0

    def reset(self):
        self.step_cnt = 0
        self.current_checkpoint_index = 0
        obs_list_of_dicts = self.game.reset()
        return [item['agent_obs'] for item in obs_list_of_dicts]

    def step(self, action_list):
        obs_list_of_dicts, original_reward, done, info = self.game.step(action_list)
        obs = [item['agent_obs'] for item in obs_list_of_dicts]
        reward = self.get_reward(original_reward)
        
        self.step_cnt += 1
        if self.step_cnt >= self.max_step:
            done = True
        
        return obs, reward, done, info

    def get_reward(self, original_reward):
        agent_reward = [0. for _ in range(self.agent_num)]
        win_signal = original_reward[1]

        checkpoint_reward = 0
        if self.current_checkpoint_index < len(self.checkpoints):
            target_x = self.checkpoints[self.current_checkpoint_index]
            current_x = self.game.agent_pos[0][0]
            if current_x >= target_x:
                checkpoint_reward = self.checkpoint_reward_value
                self.current_checkpoint_index += 1
        
        agent_reward[0] += checkpoint_reward

        if win_signal == 1:
            agent_reward[0] += 100.0
        elif win_signal == -1:
            agent_reward[0] -= 50.0
        else:
            pos_x = self.game.agent_pos[0][0]
            agent_reward[0] += (pos_x / 1000.0) * 0.1

        if win_signal == -1:
            agent_reward[1] = 100.0
        
        return agent_reward

    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)
        
    def close(self):
        # Pass the call to the underlying game object
        if hasattr(self.game, 'close') and callable(self.game.close):
            self.game.close()

    def seed(self, seed=None):
        # Pass the call to the underlying game object
        # This assumes self.game has a working .seed() method that returns a list
        self.game.set_seed(seed)
        return [seed]