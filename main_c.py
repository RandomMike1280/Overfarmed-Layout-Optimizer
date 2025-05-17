import os
import yaml
import ctypes
import time

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Grid dimensions
rows, cols = cfg['grid']['rows'], cfg['grid']['cols']

# Item parameters (order: planter, fertilizer, sprinkler, harvester)
item_types = ['auto_planter', 'auto_fertilizer', 'auto_sprinkler', 'auto_harvester']
ranges = [cfg['items'][t]['range'] for t in item_types]
costs = [cfg['items'][t].get('cost', 0) for t in item_types]

# GA parameters
ga_cfg = cfg['ga']
pop_size = ga_cfg['population_size']
top_k = ga_cfg['top_k']
mutation_rate = ga_cfg['mutation_rate']
cross_rate = ga_cfg['crossover_rate']
# Number of generations (optional in config, default 100)
generations = ga_cfg.get('generations', 100)

eval_ticks = cfg['simulation']['evaluation_ticks']
grow_time = cfg['grow']['time']

# Load the compiled DLL
lib_path = os.path.join(os.path.dirname(__file__), 'genetic_optimizer.dll')
lib = ctypes.CDLL(lib_path)

# Define function signatures
lib.init_optimizer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.init_optimizer.restype = None
lib.run_optimizer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)]
lib.run_optimizer.restype = None
lib.free_optimizer.argtypes = []
lib.free_optimizer.restype = None

# Prepare C arrays
RangesArray = ctypes.c_double * 4
CostsArray = ctypes.c_double * 4
ranges_arr = RangesArray(*ranges)
costs_arr = CostsArray(*costs)

# Output buffers
best_grid = (ctypes.c_int * (rows * cols))()
best_fitness = ctypes.c_double()

# Initialize optimizer in C
lib.init_optimizer(rows, cols, generations,
                   pop_size, top_k, eval_ticks,
                   grow_time, mutation_rate, cross_rate,
                   ranges_arr, costs_arr)
# Run optimization with timing
start = time.time()
lib.run_optimizer(best_grid, ctypes.byref(best_fitness))
end = time.time()
print(f"Optimization took {end - start:.2f} seconds")

# Display results
print(f"Best fitness = {best_fitness.value}")
# Map back to symbols
tile_map = {0: '.', 1: 'P', 2: 'F', 3: 'S', 4: 'H'}
for r in range(rows):
    row = [tile_map.get(best_grid[r*cols + c], '?') for c in range(cols)]
    print(' '.join(row))

# Cleanup
lib.free_optimizer()
