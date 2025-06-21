# --- START OF FILE avoid_rl/tracker.py ---

import os
import json
import pickle

class BestPathTracker:
    """Class to track and save the best paths during training."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_reward = float('-inf')
        self.best_successful_reward = float('-inf')
        self.best_path_data = None
        self.best_successful_path_data = None
        self.all_successful_paths = []
        
        os.makedirs(os.path.join(save_dir, 'best_paths'), exist_ok=True)
        
        self.tracking_file = os.path.join(save_dir, 'best_paths', 'path_tracking.json')
        self.load_existing_records()
    
    def load_existing_records(self):
        """Load existing records if they exist."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                    self.best_reward = data.get('best_reward', float('-inf'))
                    self.best_successful_reward = data.get('best_successful_reward', float('-inf'))
                    print(f"Loaded existing best reward: {self.best_reward}")
                    print(f"Loaded existing best successful reward: {self.best_successful_reward}")
            except Exception as e:
                print(f"Error loading existing records: {e}")
    
    def update(self, episode, reward, episode_data):
        """Update best path records."""
        updated = False
        
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_path_data = {'episode': episode, 'reward': reward, 'data': episode_data}
            
            best_path_file = os.path.join(self.save_dir, 'best_paths', 'best_overall_path.pkl')
            with open(best_path_file, 'wb') as f:
                pickle.dump(self.best_path_data, f)
            print(f"NEW BEST OVERALL REWARD: {reward:.2f} at episode {episode}")
            updated = True
        
        if episode_data['success']:
            self.all_successful_paths.append({'episode': episode, 'reward': reward, 'data': episode_data})
            
            if reward > self.best_successful_reward:
                self.best_successful_reward = reward
                self.best_successful_path_data = {'episode': episode, 'reward': reward, 'data': episode_data}
                
                best_success_file = os.path.join(self.save_dir, 'best_paths', 'best_successful_path.pkl')
                with open(best_success_file, 'wb') as f:
                    pickle.dump(self.best_successful_path_data, f)
                print(f"NEW BEST SUCCESSFUL REWARD: {reward:.2f} at episode {episode}")
                updated = True
        
        if updated:
            tracking_data = {
                'best_reward': self.best_reward,
                'best_successful_reward': self.best_successful_reward,
                'total_successful_episodes': len(self.all_successful_paths),
                'last_updated_episode': episode
            }
            with open(self.tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2)
        
        return updated
    
    def save_periodic_summary(self, episode):
        """Save periodic summary of all successful paths."""
        if len(self.all_successful_paths) > 0 and episode % 100 == 0:
            summary_file = os.path.join(self.save_dir, 'best_paths', f'successful_paths_summary_ep{episode}.json')
            summary_data = {
                'episode': episode,
                'total_successful_paths': len(self.all_successful_paths),
                'best_successful_reward': self.best_successful_reward,
                'average_successful_reward': sum(p['reward'] for p in self.all_successful_paths) / len(self.all_successful_paths),
            }
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
    
    def get_stats(self):
        """Get current statistics."""
        return {
            'best_overall_reward': self.best_reward,
            'best_successful_reward': self.best_successful_reward,
            'total_successful_episodes': len(self.all_successful_paths),
            'success_rate_recent': len([p for p in self.all_successful_paths[-100:]]) / min(100, len(self.all_successful_paths)) if self.all_successful_paths else 0
        }