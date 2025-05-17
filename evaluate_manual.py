import os
import yaml
from main import Layout, GeneticOptimizer

if __name__ == '__main__':
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    optimizer = GeneticOptimizer(config_path)

    # Define your manual grid layout below.
    # It must be a list of lists with size [rows][cols] from config:
    # - Use None for empty cells
    # - Use one of the item types in optimizer.item_types for occupied cells
    # Example: [[None, 'auto_planter', None, ...], [...], ...]
    manual_grid = [
    ['H', 'F', None, None, None, None, None, 'S'],
    [None, None, 'S', None, None, None, 'S', 'F'],
    [None, None, 'P', None, None, 'H', 'P', None],
    [None, 'S', None, None, None, None, 'F', None],
    [None, None, 'H', None, 'F', None, None, None],
    [None, None, 'H', None, None, 'S', 'P', None],
    ['F', None, None, None, None, 'S', None, 'H'],
    ['F', 'P', None, 'H', None, 'H', None, 'S']
]

    # map single-letter codes to full item types
    letter_map = {
        'P': 'auto_planter',
        'F': 'auto_fertilizer',
        'S': 'auto_sprinkler',
        'H': 'auto_harvester',
        None: None
    }
    # convert manual letter grid to full type grid
    typed_grid = [[letter_map.get(cell, None) for cell in row] for row in manual_grid]

    # Validate dimensions
    rows, cols = optimizer.rows, optimizer.cols
    if len(manual_grid) != rows or any(len(r) != cols for r in manual_grid):
        raise ValueError(f"manual_grid must be of size {rows}x{cols}")

    # Build layout and evaluate
    layout = Layout(typed_grid, optimizer.item_types)
    score = optimizer.simulate(layout)
    # compute raw harvested before cost penalty
    penalty = sum(len(layout.positions[t]) * optimizer.costs[t] for t in optimizer.item_types)
    raw_harvested = score + penalty
    print(f"Manual layout net score (harvest - cost): {score}")
    print(f"Manual layout raw harvested crops: {int(raw_harvested)}")
    crops_selling_cost = 799
    print(f"Total sold per 60 tick: {int(raw_harvested * crops_selling_cost * 60 / 60000)}")
