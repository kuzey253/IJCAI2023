import torch
import numpy as np
import sys
import os

sys.path.append('./rl_trainer/models')
from train.algo.ppo import PPO

model_loaded = False
model = None

def my_controller(observation, action_space, is_act_continuous=False):
    global model_loaded, model

    if not model_loaded:
        model = PPO()
        model.load('./olympics-integrated/run3/trained_model', episode=20)
        model_loaded = True

    if isinstance(observation, dict):
        observation = observation['agent_obs']

    observation = np.array(observation).flatten()
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

    action_index, _ = model.select_action(obs_tensor, deterministic=True)
    action_index = action_index.item()

    action = action_space[action_index]

    return action
