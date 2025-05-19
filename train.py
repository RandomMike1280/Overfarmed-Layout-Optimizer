from farming import FarmGame
from AlphaFarmer import AlphaFarmer
from summary import ModelSummary
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from MCTS import MCTS


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

# TODO: finish the rest
class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []
        self.policies = []
        self.values = []
    
    def __repr__(self):
        return f"ReplayBuffer(max_size={self.max_size})"

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]
    
    def add(self, state, policy, value):
        self.states.append(state)
        self.policies.append(policy)
        self.values.append(value)
        if len(self) > self.max_size:
            self.states.pop(0)
            self.policies.pop(0)
            self.values.pop(0)

    def sample(self, batch_size):
        idx = torch.randint(len(self), (batch_size,))
        return [torch.tensor(self[i]) for i in idx]
    
    def clear(self):
        self.states.clear()
        self.policies.clear()
        self.values.clear()

    def get_dataloader(self, batch_size, shuffle=True, pin_memory=True if torch.cuda.is_available() else False):
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory
        )
        return dataloader

class SelfPlay():
    def __init__(self, num_selfplay_games, model, MCTS, game, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  optimizer=None):
        self.num_selfplay_games = num_selfplay_games
        self.model = model
        model.to(device)
        model.eval()
        self.device = device
        self.game = game
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(10000)
        self.p_criterion = CrossEntropy
        self.v_criterion = torch.nn.MSELoss()
        self.mcts = MCTS

    def start_selfplay(self):
        for i in range(self.num_selfplay_games):
            states = []
            policies = []
            state = self.game.get_initial_state()
            turn = 1
            root = self.mcts.search(state, num_sims=800, num_threads=6)
            probs = self.mcts.get_action_probs(root)
            action = np.random.choice(range(len(probs)), p=probs)
            states.append(self.game.get_encoded_state(state))
            policies.append(probs)
            while turn < self.game.grid_size * self.game.grid_size:
                print(turn)
                state, turn = self.game.get_next_state(state, action, turn)
                root = self.mcts.set_root(root, action)
                root = self.mcts.search(state, num_sims=800, num_threads=6)
                probs = self.mcts.get_action_probs(root)
                action = np.random.choice(range(len(probs)), p=probs)
                states.append(self.game.get_encoded_state(state))
                policies.append(probs)
            value, term = self.game.get_value_and_terminated(state, turn)
            value = symlog(value)
            for i in range(len(states)):
                self.replay_buffer.add(states[i], policies[i], value)


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
    mcts = MCTS(game, model, device='cpu', cpuct=1.0)

    sp = SelfPlay(1, model=model, MCTS=mcts, game=game)
    sp.start_selfplay()


    sam = len(sp.replay_buffer)
    print(sam)