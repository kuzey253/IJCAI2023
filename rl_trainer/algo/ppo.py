# --- START OF FILE rl_trainer/algo/ppo.py ---

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import sys
from os import path
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

# We assume network.py is in the same directory (rl_trainer/algo/)
try:
    from .network import Actor, Critic, CNN_Actor, CNN_Critic
except ImportError:
    from network import Actor, Critic, CNN_Actor, CNN_Critic

device = 'cpu'

class PPO:
    def __init__(self, run_dir, obs_dim, action_dim, lr=0.0001, gamma=0.99,
                 clip_param=0.2, ppo_update_time=10, buffer_capacity=1000,
                 batch_size=32, max_grad_norm=0.5, use_cnn=False):
        
        super(PPO, self).__init__()

        # --- Hyperparameters as instance attributes ---
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_update_time = ppo_update_time
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.use_cnn = use_cnn

        # --- Use the exact network structure from your original file ---
        if self.use_cnn:
            self.actor_net = CNN_Actor(self.obs_dim, self.action_dim).to(device)
            self.critic_net = CNN_Critic(self.obs_dim).to(device)
        else:
            self.actor_net = Actor(self.obs_dim, self.action_dim).to(device)
            self.critic_net = Critic(self.obs_dim).to(device)

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)

        self.IO = run_dir is not None
        if self.IO:
            # Use the directory provided for logs
            log_path = os.path.join(run_dir, "PPO_logs_{}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            self.writer = SummaryWriter(log_path)


    def select_action(self, state, train=True):
        if not self.use_cnn:
            state = state.flatten()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state).to(device)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        # The transition is already a namedtuple from train.py, just store it
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        # --- This is your EXACT original update logic ---
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                Gt_index = Gt[index].view(-1, 1)
                
                # Use squeeze(1) as in your original file, assuming state shape is (batch, 1, features)
                V = self.critic_net(state[index].flatten(-2,-1).squeeze(1))
                delta = Gt_index - V
                advantage = delta.detach()

                action_prob = self.actor_net(state[index].flatten(-2,-1).squeeze(1)).gather(1, action[index])
                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                
                full_dist = self.actor_net(state[index].flatten(-2,-1).squeeze(1))
                dist_entropy = Categorical(full_dist).entropy()

                #action_loss = -torch.min(surr1, surr2).mean()
                action_loss = -torch.min(surr1, surr2).mean() - 0.05 * dist_entropy.mean()
                
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

                if self.IO:
                    self.writer.add_scalar('loss/policy_loss', action_loss.item(), self.training_step)
                    self.writer.add_scalar('loss/critic_loss', value_loss.item(), self.training_step)

        self.clear_buffer()

    def clear_buffer(self):
        del self.buffer[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        episode_str = str(episode) if isinstance(episode, int) else episode
        model_actor_path = os.path.join(base_path, f"actor_{episode_str}.pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, f"critic_{episode_str}.pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)

    def load(self, load_dir, episode):
        print(f'\nLoading model from: {load_dir}')
        episode_str = str(episode) if isinstance(episode, int) else episode
        model_actor_path = os.path.join(load_dir, 'trained_model', f"actor_{episode_str}.pth")
        model_critic_path = os.path.join(load_dir, 'trained_model', f"critic_{episode_str}.pth")
        
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            self.actor_net.load_state_dict(torch.load(model_actor_path, map_location=device))
            self.critic_net.load_state_dict(torch.load(model_critic_path, map_location=device))
            print("✓ Model loaded successfully!")
        else:
            sys.exit(f'ERROR: Model not found at specified path!')