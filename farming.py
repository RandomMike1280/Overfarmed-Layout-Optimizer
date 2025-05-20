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
        if action == 0:
            new_state[row][col] = -1
        else:
            new_state[row][col] = action
        return new_state, turn + 1

    def get_valid_moves(self, state):
        return np.ones(self.action_size)

    def check_win(self, state, action):
        # win if all cells are filled
        return np.all(state)

    def change_perspective(self, state, player):
        return state

    def is_same_state(self, state1, state2):
        # check if 2 states are the same
        # they're the same if they're identical when rotated, mirrored, flipped in any way
        
        if state1.shape != state2.shape:
            return False

        # Check original state1 and its 3 further rotations
        current_s1_variant = state1
        for _ in range(4):  # Covers 0, 90, 180, 270 degree rotations
            if np.array_equal(current_s1_variant, state2):
                return True
            current_s1_variant = np.rot90(current_s1_variant)

        # Check a flipped version of state1 (e.g., left-right flip) and its 3 further rotations
        current_s1_variant = np.fliplr(state1) 
        for _ in range(4):  # Covers 0, 90, 180, 270 degree rotations of the flipped state
            if np.array_equal(current_s1_variant, state2):
                return True
            current_s1_variant = np.rot90(current_s1_variant)
            
        return False

    def get_value_and_terminated(self, state, turn):
        if turn != self.grid_size * self.grid_size:
            return 0, False
        optimizer = GeneticOptimizer('config.yaml')
        manual_grid = state
        letter_map = {
            1: 'auto_planter',
            2: 'auto_fertilizer',
            3: 'auto_sprinkler',
            4: 'auto_harvester',
            -1: None
        }

        typed_grid = [[letter_map.get(cell, None) for cell in row] for row in manual_grid]

        # Validate dimensions
        rows, cols = self.grid_size, self.grid_size
        if len(manual_grid) != rows or any(len(r) != cols for r in manual_grid):
            raise ValueError(f"manual_grid must be of size {rows}x{cols}")

        # Build layout and evaluate
        layout = Layout(typed_grid, optimizer.item_types)
        score = optimizer.simulate(layout)
        return score, True
    
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
    state, turn = game.get_next_state(state, 1, 0)
    print(state)
    print(game.get_encoded_state(state))