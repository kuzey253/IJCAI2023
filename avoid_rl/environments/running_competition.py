import math
import numpy as np
import pygame
from olympics_engine.generator import create_scenario
from scenario.running_competition import Running_competition as Running


class RunningCompetitionEnv:
    def __init__(self, map_id):
        """
        Initializes the environment with a specific, predetermined map_id.
        :param map_id: An integer (e.g., 1-10) corresponding to a hardcoded map.
        """
        # Load the base configuration for the game type
        Gamemap_config = create_scenario("running-competition")
        
        # Instantiate the underlying game, passing the map_id to it
        self.game = Running(meta_map=Gamemap_config, map_id=map_id)
        
        self.agent_num = self.game.agent_num
        
        # Initialize map-dependent properties
        # We need to get the full map dictionary from the game object after it's built the map
        self._initialize_map_properties(self.game.map)

        # The rest of the properties
        self.max_step = self.game.max_step if hasattr(self.game, 'max_step') else 1000
        self.checkpoints = [0.25, 0.5, 0.75, 0.95]
        self.checkpoint_reward_value = 25.0
        self.debug_target_info = None

    def _initialize_map_properties(self, gamemap):
        self.track_path = self.generate_track_path(gamemap)
        self.track_lengths = [np.linalg.norm(self.track_path[i+1] - self.track_path[i]) for i in range(len(self.track_path)-1)]
        self.total_track_length = sum(self.track_lengths)
        if self.total_track_length == 0:
            print("ERROR: Track has zero length!")
            # Handle error, maybe by setting a default length
            self.total_track_length = 1.0

    def seed(self, seed=None):
        # The game object already exists, so we just need to seed it
        self.game.set_seed(seed)
        return [seed]
    
    def reset(self):
        self.step_cnt = 0
        self.current_checkpoint_index = 0
        self.is_first_step = True
        obs_list = self.game.reset()
        self.last_progress = self.get_agent_progress()
        return obs_list

    def step(self, action_list):
        # We need the action for the efficiency calculation
        agent_action = action_list[0] # TODO: Handle multi-agent actions reward if needed
        
        obs, original_reward, done, info_from_game = self.game.step(action_list)

        if not isinstance(info_from_game, dict):
            info = {}
        else:
            info = info_from_game
        # ----------------------

        reward = self.get_reward(original_reward, agent_action)

        if self.is_first_step:
            self.is_first_step = False

        self.step_cnt += 1
        if self.step_cnt >= self.max_step:
            done = True
        
        # Add our custom info to the (now guaranteed to be) dictionary
        if done:
            info['final_progress'] = self.get_agent_progress()
            info['win_signal'] = original_reward[1]

        return obs, reward, done, info

    def get_arrow_alignment_reward(self, agent_idx=0):
        # 1. Find all potential arrow line objects (same as before)
        all_arrow_lines = [
            obj for obj in self.game.map['objects'] 
            if (hasattr(obj, 'type') and 'cross' in obj.type.lower() and 
                hasattr(obj, 'color') and obj.color in ['grey', 'light blue'])
        ]

        # --- NEW: GROUPING LOGIC ---
        # Group close-by lines into single "chevrons"
        chevrons = []
        processed_indices = set()
        
        for i in range(len(all_arrow_lines)):
            if i in processed_indices:
                continue
            
            line1 = all_arrow_lines[i]
            center1 = np.mean(line1.init_pos, axis=0)
            
            # Find a partner line that is very close
            partner_idx = -1
            min_dist = float('inf')
            
            for j in range(i + 1, len(all_arrow_lines)):
                if j in processed_indices:
                    continue
                line2 = all_arrow_lines[j]
                center2 = np.mean(line2.init_pos, axis=0)
                dist = np.linalg.norm(center1 - center2)
                
                # A threshold to consider two lines part of the same chevron (e.g., < 30 pixels apart)
                if dist < 30.0 and dist < min_dist:
                    min_dist = dist
                    partner_idx = j
            
            if partner_idx != -1:
                # Found a pair, group them. The chevron's position is the average of the two centers.
                # The chevron's direction is the average of the two line directions.
                partner_line = all_arrow_lines[partner_idx]
                center2 = np.mean(partner_line.init_pos, axis=0)
                
                vec1 = np.array(line1.init_pos[1]) - np.array(line1.init_pos[0])
                vec2 = np.array(partner_line.init_pos[1]) - np.array(partner_line.init_pos[0])
                
                chevron_center = (center1 + center2) / 2.0
                chevron_vector = (vec1 / (np.linalg.norm(vec1) + 1e-6)) + (vec2 / (np.linalg.norm(vec2) + 1e-6))
                
                chevrons.append({'center': chevron_center, 'vector': chevron_vector})
                processed_indices.add(i)
                processed_indices.add(partner_idx)
            else:
                # This line has no close partner, treat it as a single-line arrow
                vec = np.array(line1.init_pos[1]) - np.array(line1.init_pos[0])
                chevrons.append({'center': center1, 'vector': vec})
                processed_indices.add(i)

        # --- The rest of the logic now uses the 'chevrons' list ---
        if self.is_first_step:
            if chevrons:
                print(f"DEBUG: Found {len(chevrons)} guide chevrons (from {len(all_arrow_lines)} lines). Arrow-following reward is ACTIVE.")
            else:
                print("DEBUG: No guide chevrons found. Arrow-following reward is INACTIVE.")
        
        if not chevrons:
            return 0.0

        agent_pos = np.array(self.game.agent_pos[agent_idx])
        agent_vel = np.array(self.game.agent_v[agent_idx])
        
        # 2. Find the closest CHEVRON to the agent
        closest_chevron = None
        min_dist_sq = float('inf')
        for chevron in chevrons:
            dist_sq = np.sum((agent_pos - chevron['center'])**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_chevron = chevron

        # 3. Calculate alignment with the CHEVRON's direction
        if closest_chevron is not None:
            self.debug_target_info = closest_chevron 
            # Normalize the agent's velocity and the chevron's vector
            arrow_vector = closest_chevron['vector']
            
            norm_agent_vel = agent_vel / (np.linalg.norm(agent_vel) + 1e-6)
            norm_arrow_vec = arrow_vector / (np.linalg.norm(arrow_vector) + 1e-6)
            
            alignment = np.dot(norm_agent_vel, norm_arrow_vec)
            return alignment * 0.5
        else:
            self.debug_target_info = None
            
        return 0.0

    # --- MODIFIED get_reward FUNCTION ---
    def get_reward(self, original_reward, agent_action):
        agent_reward = [0. for _ in range(self.agent_num)]
        win_signal = original_reward[1]

        if win_signal == 1:
            agent_reward[0] = 100.0
        elif win_signal == -1:
            agent_reward[0] = -100.0
        else:
            # --- ADD THE NEW ARROW REWARD ---
            arrow_reward = self.get_arrow_alignment_reward() * 2.0
            agent_reward[0] += arrow_reward

            # --- Keep the other rewards, but maybe scale them slightly differently ---
            current_progress = self.get_agent_progress()
            progress_delta = current_progress - self.last_progress
            agent_reward[0] += progress_delta * 50.0

            force_used = abs(agent_action[0]) / 200.0
            epsilon = 1e-6
            
            # We can slightly reduce the efficiency reward now that we have the arrow guidance
            efficiency_reward = (progress_delta / (force_used + epsilon)) * 100.0
            agent_reward[0] += efficiency_reward

            distance_from_track = self.get_distance_from_track()
            off_track_penalty = (distance_from_track / 100.0) * 0.005
            agent_reward[0] -= off_track_penalty
            
            if self.current_checkpoint_index < len(self.checkpoints):
                target_progress = self.checkpoints[self.current_checkpoint_index]
                if current_progress >= target_progress:
                    agent_reward[0] += self.checkpoint_reward_value
                    self.current_checkpoint_index += 1

        self.last_progress = self.get_agent_progress()
        if win_signal == -1:
            agent_reward[1] = 100.0
        return agent_reward
    
        # --- NEW: Helper methods for progress calculation ---
    def get_agent_progress(self, agent_idx=0):
        agent_pos = np.array(self.game.agent_pos[agent_idx])
        proj_info = self.find_closest_segment_and_projection(agent_pos)
        progress_to_segment = sum(self.track_lengths[:proj_info['segment_index']])
        current_segment_length = self.track_lengths[proj_info['segment_index']]
        progress_on_segment = proj_info['progress_on_segment'] * current_segment_length
        total_progress_dist = progress_to_segment + progress_on_segment
        return total_progress_dist / self.total_track_length if self.total_track_length > 0 else 0

    def get_distance_from_track(self, agent_idx=0):
        agent_pos = np.array(self.game.agent_pos[agent_idx])
        proj_info = self.find_closest_segment_and_projection(agent_pos)
        return proj_info['distance']

    def render(self, *args, **kwargs):
        result = self.game.render(*args, **kwargs)

        if self.debug_target_info is not None:
            screen = pygame.display.get_surface()

            if screen:
                agent_pos = self.game.agent_pos[0]
                target_pos = self.debug_target_info['center']
                
                # --- THIS IS THE FIX ---
                # Assume a direct 1-to-1 mapping from game coordinates to screen pixels.
                # We just need to convert them to integer tuples for pygame.draw.
                agent_screen_pos = (int(agent_pos[0]), int(agent_pos[1]))
                target_screen_pos = (int(target_pos[0]), int(target_pos[1]))

                pygame.draw.line(screen, (255, 105, 180), # Hot pink color
                                 agent_screen_pos, 
                                 target_screen_pos, 2)
                
                pygame.display.flip()

        return result


    def close(self):
        if hasattr(self.game, 'close') and callable(self.game.close):
            self.game.close()

    def seed(self, seed=None):
        self.game.set_seed(seed)
        return [seed]
    
    
    def find_closest_segment_and_projection(self, agent_pos):
        best_proj_info = {
            'distance': float('inf'), 'projection': None, 'segment_index': -1, 'progress_on_segment': 0.0
        }
        for i in range(len(self.track_path) - 1):
            p1, p2 = self.track_path[i], self.track_path[i+1]
            segment_vec, agent_vec = p2 - p1, agent_pos - p1
            segment_len_sq = np.dot(segment_vec, segment_vec)
            if segment_len_sq == 0: continue
            t = np.dot(agent_vec, segment_vec) / segment_len_sq
            projection = p1 + np.clip(t, 0, 1) * segment_vec
            distance = np.linalg.norm(agent_pos - projection)
            if distance < best_proj_info['distance']:
                best_proj_info.update({
                    'distance': distance, 'projection': projection, 'segment_index': i, 'progress_on_segment': np.clip(t, 0, 1)
                })
        return best_proj_info
    

    # in environments/running_competition.py, inside RunningCompetitionEnv class

    def generate_track_path(self, gamemap):
        # --- MODIFIED: FIND ALL BOUNDARY OBJECTS (WALLS AND ARCS) ---
        boundary_objects = [
            obj for obj in gamemap['objects'] 
            if hasattr(obj, 'type') and ('wall' in obj.type.lower() or 'arc' in obj.type.lower())
        ]
        
        all_points = []
        for obj in boundary_objects:
            # --- HANDLE WALLS ---
            if 'wall' in obj.type.lower():
                vertex_data = None
                if hasattr(obj, 'init_pos'): vertex_data = obj.init_pos
                elif hasattr(obj, 'initial_position'): vertex_data = obj.initial_position
                
                if vertex_data and isinstance(vertex_data, list) and len(vertex_data) > 0 and isinstance(vertex_data[0], list):
                    all_points.extend(vertex_data)
            
            # --- NEW: HANDLE ARCS by sampling points ---
            elif 'arc' in obj.type.lower():
                # We need to extract arc parameters and sample points along its curve
                # Arc objects have 'init_pos' = [center_x, center_y, width, height], 'start_radian', 'end_radian'
                if hasattr(obj, 'init_pos') and hasattr(obj, 'start_radian') and hasattr(obj, 'end_radian'):
                    center_x, center_y, width, height = obj.init_pos
                    # Convert degrees to radians if necessary (assuming the engine uses degrees)
                    start_rad = np.deg2rad(obj.start_radian)
                    end_rad = np.deg2rad(obj.end_radian)
                    
                    # Sample 20 points along the arc
                    for rad in np.linspace(start_rad, end_rad, 20):
                        x = center_x + (width / 2) * np.cos(rad)
                        y = center_y + (height / 2) * np.sin(rad)
                        all_points.append([x, y])

        if len(all_points) < 10: # Increased threshold for complex maps
            print(f"WARNING: Not enough boundary vertices ({len(all_points)} found) for map. Using fallback path.")
            return [np.array([100, 350]), np.array([900, 350])]

        np_points = np.array(all_points)
        
        # --- The Nearest-Neighbor algorithm from before is perfect for this ---
        
        # 1. Find a reliable starting point (e.g., minimum X value)
        start_idx = np.argmin(np_points[:, 0])
        start_point = np_points[start_idx]
        
        # 2. Iteratively build an ordered list of boundary points
        ordered_boundary = [start_point.tolist()]
        remaining_indices = list(range(len(np_points)))
        remaining_indices.pop(start_idx)

        current_point = start_point
        while remaining_indices:
            closest_dist_sq = float('inf')
            next_idx_in_list = -1
            
            for i, point_idx in enumerate(remaining_indices):
                dist_sq = np.sum((current_point - np_points[point_idx])**2)
                if dist_sq < closest_dist_sq:
                    closest_dist_sq = dist_sq
                    next_idx_in_list = i
            
            original_point_idx = remaining_indices.pop(next_idx_in_list)
            current_point = np_points[original_point_idx]
            ordered_boundary.append(current_point.tolist())

        # 3. Create the centerline by averaging opposite sides of the boundary trace
        num_boundary_points = len(ordered_boundary)
        half_len = num_boundary_points // 2
        
        if half_len < 2:
            print(f"WARNING: Path generated from boundary is too short ({num_boundary_points} points). Using fallback.")
            return [np.array([100, 350]), np.array([900, 350])]

        path_side1 = np.array(ordered_boundary[:half_len])
        path_side2 = np.array(ordered_boundary[half_len:2*half_len][::-1])

        min_len = min(len(path_side1), len(path_side2))
        path_side1 = path_side1[:min_len]
        path_side2 = path_side2[:min_len]
        
        centerline = (path_side1 + path_side2) / 2.0
        path = [np.array(p) for p in centerline]

        print(f"DEBUG: Successfully generated a {len(path)}-segment path from {len(all_points)} boundary points.")
        return path