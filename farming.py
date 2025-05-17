import numpy as np
from game import Game
from main import Layout, GeneticOptimizer

class FarmGame(Game):
    def __init__(self, grid_size:int):
        self.grid_size = grid_size
        self.action_size = 5

    def __repr__(self):
        return "FarmGame"
    
    def get_initial_state(self):
        return np.zeros((self.grid_size, self.grid_size))

    def get_next_state(self, state, action, turn):
        new_state = state.copy()
        row = turn // self.grid_size
        col = turn % self.grid_size
        new_state[row][col] = action
        return new_state, turn + 1

    def get_valid_moves(self, state):
        return np.ones(self.action_size)

    def check_win(self, state, action):
        # win if all cells are filled
        return np.all(state)

    def change_perspective(self, state, player):
        return state

    def get_value_and_terminated(self, state, action):
        manual_grid = state
        letter_map = {
            1: 'auto_planter',
            2: 'auto_fertilizer',
            3: 'auto_sprinkler',
            4: 'auto_harvester',
            -1: None
        }
        # convert manual letter grid to full type grid
        typed_grid = [[letter_map.get(cell, None) for cell in row] for row in manual_grid]

        # Validate dimensions
        rows, cols = self.grid_size, self.grid_size
        if len(manual_grid) != rows or any(len(r) != cols for r in manual_grid):
            raise ValueError(f"manual_grid must be of size {rows}x{cols}")

        # Build layout and evaluate
        layout = Layout(typed_grid, optimizer.item_types)
        score = optimizer.simulate(layout)
        return score
    
    def get_encoded_state(self, state):
        # one hot encode state
        encoded_state = np.zeros((self.action_size+1, self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                v = int(state[i][j])
                encoded_state[v][i][j] = 1
        return encoded_state

if __name__ == "__main__":
    game = FarmGame(8)
    state = game.get_initial_state()
    print(state)
    print(game.get_valid_moves(state))
    print(game.get_next_state(state, 1, 0))
    print(game.get_encoded_state(state))