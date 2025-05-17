from farming import FarmGame
from AlphaFarmer import AlphaFarmer
from summary import ModelSummary
import torch

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

# TODO: finish the rest
