# --- START OF FILE olympics_engine/scenario/running.py ---

from olympics_engine.core import OlympicsBase
from olympics_engine.viewer import Viewer, debug
import time
import pygame
import sys

class Running(OlympicsBase):
    def __init__(self, map, seed = None):
        self.minimap_mode = map['obs_cfg'].get('minimap', False)

        # This is the line we added to fix the previous error
        self.meta_map = map

        super(Running, self).__init__(map, seed)

        self.game_name = 'running'
        self.tau = 0.1
        self.gamma = 1
        self.wall_restitution = 1
        self.circle_restitution = 1
        self.max_step = map['env_cfg'].get('max_step', 500)
        
        self.draw_obs = True
        self.show_traj = True

    def reset(self):
        self.set_seed()
        self.init_state()
        self.step_cnt = 0
        self.done = False

        self.viewer = Viewer(self.view_setting)
        self.display_mode=False

        init_obs = self.get_obs()
        return self._build_from_raw_obs(init_obs)

    def get_reward(self):
        # Returns a list: [total_reward, win_signal]
        # win_signal: 1 if agent 0 wins, -1 if agent 1 (opponent) wins, 0 otherwise
        if self.agent_list[0].finished:
            return [100.0, 1]
        elif self.agent_list[1].finished:
            return [-100.0, -1]
        
        return [0.0, 0]

    def is_terminal(self):
        if self.step_cnt >= self.max_step:
            return True

        if self.agent_list[0].finished or self.agent_list[1].finished:
            return True

        return False

    def step(self, actions_list):
        previous_pos = self.agent_pos
        self.stepPhysics(actions_list, self.step_cnt)
        self.speed_limit()
        self.cross_detect(previous_pos, self.agent_pos)

        self.step_cnt += 1
        
        original_reward = self.get_reward()
        done = self.is_terminal()
        obs_next = self.get_obs()
        self.change_inner_state()
        
        output_obs_next = self._build_from_raw_obs(obs_next)
        
        return output_obs_next, original_reward, done, ''

    def _build_from_raw_obs(self, obs):
        # The base game returns a list of dictionaries as specified by the engine's standard
        return [{"agent_obs": obs[0], "id":"team_0"}, 
                {"agent_obs": obs[1], "id":"team_1"}]

    def render(self, info=None):
        if not self.display_mode:
            self.viewer.set_mode()
            self.display_mode=True

        self.viewer.draw_background()
        for w in self.map['objects']:
            self.viewer.draw_map(w)

        self.viewer.draw_ball(self.agent_pos, self.agent_list)

        if self.draw_obs:
            self.viewer.draw_obs(self.obs_boundary, self.agent_list)
            if len(self.obs_list) > 0:
                self.viewer.draw_view(self.obs_list, self.agent_list, leftmost_x=500, upmost_y=10, gap = 100)

        if self.show_traj:
            self.get_trajectory()
            self.viewer.draw_trajectory(self.agent_record, self.agent_list)

        self.viewer.draw_direction(self.agent_pos, self.agent_accel)
        debug('Step: ' + str(self.step_cnt), x=30)
        if info is not None:
            debug(info, x=100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()