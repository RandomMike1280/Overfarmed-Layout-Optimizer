from farming import FarmGame
from AlphaFarmer import AlphaFarmer
from summary import ModelSummary
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Proximal Policy Optimization
"""
L_CLIP = E^_t[min(r_t(theta) * A^_t, clip(r_t(theta), 1-epsilon, 1+epsilon)*A^_t)]

r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
A^_t = sum(l = 0 -> T-t-1, (gamma * lambda)**l * delta_(t+l))

delta_t = r_t + gamma * V(s_t+1) - V(s_t)
r_t is the reward at time (t), (all 0s and only becomes non-zero at the final state)

L_VF = E^_t[(V_theta(s_t) - V_t^target)**2]

# Entropy Bonus:
S[pi_theta](s_t) = -E^_t[sum(pi_theta(a_t|s_t) * log(pi_theta(a_t|s_t)))]

total_loss = -L_CLIP(theta) + c_1 * L_VF(theta) - c_2 * S[pi_theta](s_t) 
"""

def test():
    game = FarmGame(8)
    model = AlphaFarmer(6, (8, 8), 5)
    ModelSummary(model, input_size=(6, 8, 8))
    input = game.get_initial_state()
    encoded_input = torch.tensor(game.get_encoded_state(input), dtype=torch.float32).unsqueeze(0)
    output = model(encoded_input)
    print(output)

def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def CrossEntropy(x, y):
    return -torch.sum(y * torch.log(x + 1e-9)) / x.size(0)

def ValueLoss(prediction, target):
    return torch.mean((prediction - target) ** 2)

def PolicyLoss(prediction, prediction_old, advantage):
    epsilon = 0.2
    ratio = prediction / prediction_old
    clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
    return torch.mean(loss)

def EntropyBonus(policy):
    return -torch.sum(policy * torch.log(policy + 1e-9)) / policy.size(0)

def GAE(rewards, next_values, values, gamma=0.99, lambd=0.01):
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = 0
        else:
            next_val = next_values[t+1]
            
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lambd * gae
        advantages[t] = gae
        
    return advantages

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x),)

class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []
        self.policies = []
        self.values = []
        self.reward = []
    
    def __repr__(self):
        return f"ReplayBuffer(max_size={self.max_size})"

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx], self.reward[idx]
    
    def add(self, state, policy, value, reward):
        self.states.append(state)
        self.policies.append(policy)
        self.values.append(value)
        self.reward.append(reward)
        if len(self) > self.max_size:
            self.states.pop(0)
            self.policies.pop(0)
            self.values.pop(0)
            self.reward.pop(0)

    def sample(self, batch_size):
        idx = torch.randint(len(self), (batch_size,))
        return [torch.tensor(self[i]) for i in idx]
    
    def clear(self):
        self.states.clear()
        self.policies.clear()
        self.values.clear()
        self.reward.clear()

    def get_dataloader(self, batch_size, shuffle=True, pin_memory=True if torch.cuda.is_available() else False):
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory
        )
        return dataloader

class SelfPlay():
    def __init__(self, num_selfplay_games, model, game, jit=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  optimizer=None):
        self.num_selfplay_games = num_selfplay_games
        self.model = model
        model.to(device)
        self.old_model = model
        self.old_model.to(device)
        if jit:
            try:
                self.model = torch.jit.script(model)
                self.old_model = torch.jit.script(model)
                print("Successfully jitted the model")
            except Exception as e:
                print(f"Failed to jit script the model, error: {e}")
        self.model.eval()
        self.old_model.eval()
        self.device = device
        self.game = game
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(10000)
        # PPO hyperparameters
        self.c1 = 0.5  # Value loss coefficient
        self.c2 = 0.01  # Entropy bonus coefficient
        self.epsilon = 0.2  # Clipping parameter

    def start_selfplay(self):
        for i in range(self.num_selfplay_games):
            state = self.game.get_initial_state()
            turn = 1
            encoded_state = torch.tensor(self.game.get_encoded_state(state), dtype=torch.float32).to(self.device).unsqueeze(0)
            policy, value = self.model(encoded_state)
            probs = policy.detach().cpu().numpy()[0]
            if sum(probs) != 1:
                probs = Softmax(probs)
            action = np.random.choice(range(len(probs)), p=probs)
            self.replay_buffer.add(state, policy, value, 0)
            while turn < self.game.grid_size * self.game.grid_size:
                state, turn = self.game.get_next_state(state, action, turn)
                encoded_state = torch.tensor(self.game.get_encoded_state(state), dtype=torch.float32).to(self.device).unsqueeze(0)
                policy, value = self.model(encoded_state)
                probs = policy.detach().cpu().numpy()[0]
                if sum(probs) != 1:
                    probs = Softmax(probs)
                action = np.random.choice(range(len(probs)), p=probs)
                self.replay_buffer.add(state, policy, value, 0)
            reward, term = self.game.get_value_and_terminated(state, turn)
            reward = symlog(reward)
            self.replay_buffer.add(state, policy, value, reward)
    
    def optimize(self, num_epoch=2, batch_size=64):
        # Proximal Policy Optimization
        self.model.train()
        dataloader = self.replay_buffer.get_dataloader(batch_size)
        
        for epoch in range(num_epoch):
            print(f"Epoch {epoch + 1}/{num_epoch}")
            for states, old_policies, old_values, rewards in dataloader:
                states = torch.tensor(states).to(self.device)
                old_policies = torch.tensor(old_policies).to(self.device)
                old_values = torch.tensor(old_values).to(self.device)
                rewards = torch.tensor(rewards).to(self.device)
                
                # Get current policy and value predictions
                self.optimizer.zero_grad()
                policies, values = self.model(states)
                
                # Calculate advantages using GAE
                next_values = torch.cat([values[1:], torch.zeros(1).to(self.device)])
                advantages = GAE(rewards, next_values, values)
                
                # Calculate PPO losses
                policy_loss = PolicyLoss(policies, old_policies, advantages)
                value_loss = ValueLoss(values, rewards)
                entropy_bonus = EntropyBonus(policies)
                
                # Total loss as per PPO formula: -L_CLIP + c_1 * L_VF - c_2 * S
                total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy_bonus
                
                # Backpropagation
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
        # Update old model with current model parameters
        self.old_model.load_state_dict(self.model.state_dict())
        self.replay_buffer.clear()
        self.model.eval()
        return self.model, self.optimizer

if __name__ == '__main__':
    # rb = ReplayBuffer(10000)
    # print(rb)
    # rb.add([1], [2], [3])
    # dt = rb.get_dataloader(batch_size=1)
    # print(dt)

    # for states, policies, value in dt:
    #     print(states)
    #     print(policies)
    #     print(value)
    #     break

    game = FarmGame(43)
    model = AlphaFarmer(6, (43, 43), 5)

    sp = SelfPlay(1, model=model, game=game)
    sp.start_selfplay()


    sam = len(sp.replay_buffer)
    print(sam)