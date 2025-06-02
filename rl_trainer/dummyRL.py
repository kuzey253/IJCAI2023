import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a very simple dummy environment
class DummyEnv:
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2

    def reset(self):
        return np.random.randn(self.state_dim)

    def step(self, action):
        next_state = np.random.randn(self.state_dim)
        reward = np.random.rand()
        done = np.random.rand() > 0.95  # 5% chance of episode ending
        return next_state, reward, done, {}

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        return torch.softmax(self.fc2(x), dim=-1)

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        return self.fc2(x)

# Basic PPO Training
def train_simple_ppo():
    env = DummyEnv()

    actor = Actor(env.state_dim, env.action_dim)
    critic = Critic(env.state_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    gamma = 0.99
    eps_clip = 0.2

    for episode in range(10):  # Just 10 episodes
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())
            reward_tensor = torch.FloatTensor([reward])

            # Calculate advantage
            value = critic(state_tensor)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_value = critic(next_state_tensor)

            advantage = reward_tensor + gamma * next_value - value

            # Actor Loss (PPO Clip Loss)
            new_probs = actor(state_tensor)
            new_dist = torch.distributions.Categorical(new_probs)
            new_log_prob = new_dist.log_prob(action)

            ratio = (new_log_prob - log_prob).exp()
            actor_loss = -torch.min(ratio * advantage,
                                    torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage).mean()

            # Critic Loss
            target = reward_tensor + gamma * next_value
            critic_loss = (target - value).pow(2).mean()

            # Update Actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update Critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            state = next_state

        print(f"Episode {episode+1} completed.")

    # After training, save models
    torch.save(actor.state_dict(), 'simple_actor.pth')
    torch.save(critic.state_dict(), 'simple_critic.pth')
    print("Models saved: simple_actor.pth and simple_critic.pth")

# Run the training
if __name__ == "__main__":
    train_simple_ppo()
