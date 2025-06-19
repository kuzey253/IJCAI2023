# --- START OF FILE avoid_rl/utils.py ---

import numpy as np

def check_convergence(episode, record_win, record_reward, min_episodes=200, win_rate_threshold=0.95, reward_std_threshold=15.0, improvement_threshold=5.0):
    """
    Check for convergence based on stable, high win rates and rewards.
    """
    if episode < min_episodes or len(record_win) < 100:
        return False

    current_win_rate = sum(record_win) / len(record_win)
    if current_win_rate < win_rate_threshold:
        return False

    reward_array = np.array(list(record_reward))
    current_reward_std = np.std(reward_array)
    if current_reward_std > reward_std_threshold:
        return False

    first_half_avg = np.mean(reward_array[:50])
    second_half_avg = np.mean(reward_array[50:])
    improvement = second_half_avg - first_half_avg
    
    if abs(improvement) > improvement_threshold:
        return False

    print("\n" + "="*50)
    print(f"CONVERGENCE DETECTED at episode {episode}!")
    print(f"  - Win Rate (last 100 ep): {current_win_rate:.3f} >= {win_rate_threshold}")
    print(f"  - Reward Std Dev (last 100 ep): {current_reward_std:.2f} <= {reward_std_threshold}")
    print(f"  - Reward Plateau (abs change): {abs(improvement):.2f} <= {improvement_threshold}")
    print("="*50 + "\n")
    return True