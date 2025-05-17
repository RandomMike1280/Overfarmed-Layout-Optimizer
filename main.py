import os
import sys
import yaml
import random
import time
import copy
import json

class Layout:
    def __init__(self, grid, item_types):
        self.grid = grid  # 2D list of type or None
        self.positions = {t: [] for t in item_types}
        for i, row in enumerate(grid):
            for j, t in enumerate(row):
                if t in self.positions:
                    self.positions[t].append((i, j))

    def print(self, rows, cols):
        symbols = {
            None: '.',
            'auto_planter': 'P',
            'auto_fertilizer': 'F',
            'auto_sprinkler': 'S',
            'auto_harvester': 'H'
        }
        for i in range(rows):
            line = []
            for j in range(cols):
                line.append(symbols.get(self.grid[i][j], '?'))
            print(' '.join(line))

    def to_dict(self):
        return self.positions

class GeneticOptimizer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.rows = self.cfg['grid']['rows']
        self.cols = self.cfg['grid']['cols']
        items_cfg = self.cfg['items']
        self.item_types = ['auto_planter','auto_fertilizer','auto_sprinkler','auto_harvester']
        self.ranges = {t: items_cfg[t]['range'] for t in self.item_types}
        self.costs = {t: items_cfg[t].get('cost', 0) for t in self.item_types}
        grow_cfg = self.cfg['grow']
        self.grow_time = grow_cfg['time']
        self.fert_bonus = grow_cfg['fertilizer_bonus']
        self.water_bonus = grow_cfg['water_bonus']
        ga_cfg = self.cfg['ga']
        self.pop_size = ga_cfg['population_size']
        self.mutation_rate = ga_cfg['mutation_rate']
        self.cross_rate = ga_cfg['crossover_rate']
        self.top_k = ga_cfg['top_k']
        sim_cfg = self.cfg['simulation']
        self.eval_ticks = sim_cfg['evaluation_ticks']
        self.population = []
        self.best_layout = None
        self.best_fitness = -1

        # precompute all cells
        self.all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def generate_random_layout(self):
        grid = [[random.choice([None] + self.item_types) for _ in range(self.cols)]
                for _ in range(self.rows)]
        return Layout(grid, self.item_types)

    def initialize_population(self):
        return [self.generate_random_layout() for _ in range(self.pop_size)]

    def simulate(self, layout):
        # grid of crops: each cell None or dict with 'progress'
        crops = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        harvested = 0
        for tick in range(self.eval_ticks):
            # planter
            for r in range(self.rows):
                for c in range(self.cols):
                    if layout.grid[r][c] == 'auto_planter':
                        side = self.ranges['auto_planter']
                        half = side // 2
                        for i in range(max(0, r-half), min(self.rows, r-half+side)):
                            for j in range(max(0, c-half), min(self.cols, c-half+side)):
                                # only plant on empty cells (no item here)
                                if crops[i][j] is None and layout.grid[i][j] is None:
                                    crops[i][j] = {
                                        'progress': 0.0,
                                        'fertilized': False,
                                        'watered': False,
                                        'time_required': self.grow_time
                                    }
            # fertilizer
            for r in range(self.rows):
                for c in range(self.cols):
                    t = layout.grid[r][c]
                    if t == 'auto_fertilizer':
                        side = self.ranges[t]; half = side // 2
                        for i in range(max(0, r-half), min(self.rows, r-half+side)):
                            for j in range(max(0, c-half), min(self.cols, c-half+side)):
                                crop = crops[i][j]
                                if crop is not None and not crop['fertilized']:
                                    crop['fertilized'] = True
                                    crop['time_required'] = (
                                        self.grow_time * 0.5 if crop['watered'] else self.grow_time * 0.75
                                    )
                    if t == 'auto_sprinkler':
                        side = self.ranges[t]; half = side // 2
                        for i in range(max(0, r-half), min(self.rows, r-half+side)):
                            for j in range(max(0, c-half), min(self.cols, c-half+side)):
                                crop = crops[i][j]
                                if crop is not None and not crop['watered']:
                                    crop['watered'] = True
                                    crop['time_required'] = (
                                        self.grow_time * 0.5 if crop['fertilized'] else self.grow_time * 0.75
                                    )
            # growth
            for i in range(self.rows):
                for j in range(self.cols):
                    crop = crops[i][j]
                    if crop is not None:
                        crop['progress'] += (1.0 / crop['time_required'])
            # harvester
            for r in range(self.rows):
                for c in range(self.cols):
                    if layout.grid[r][c] == 'auto_harvester':
                        side = self.ranges['auto_harvester']
                        half = side // 2
                        for i in range(max(0, r-half), min(self.rows, r-half+side)):
                            for j in range(max(0, c-half), min(self.cols, c-half+side)):
                                crop = crops[i][j]
                                if crop is not None and crop['progress'] >= 1.0:
                                    harvested += 1
                                    crops[i][j] = None
        # apply placement cost penalty
        cost_penalty = sum(len(layout.positions[t]) * self.costs[t] for t in self.item_types)
        return harvested - cost_penalty

    def select_parents(self, fitnesses):
        # ensure non-negative weights for selection
        weights = [max(0.0, f) for f in fitnesses]
        total = sum(weights)
        if total <= 0:
            # if no positive fitness, pick random parents
            return random.sample(self.population, 2)
        # select based on positive fitness weights
        parents = random.choices(self.population, weights=weights, k=2)
        return parents

    def crossover(self, p1, p2):
        # uniform crossover on grid
        grid = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(p1.grid[i][j] if random.random() < self.cross_rate else p2.grid[i][j])
            grid.append(row)
        return Layout(grid, self.item_types)

    def mutate(self, layout):
        # mutate each cell's type
        for i in range(self.rows):
            for j in range(self.cols):
                if random.random() < self.mutation_rate:
                    layout.grid[i][j] = random.choice([None] + self.item_types)
        # update positions
        layout.positions = {t: [] for t in self.item_types}
        for i in range(self.rows):
            for j in range(self.cols):
                t = layout.grid[i][j]
                if t in layout.positions:
                    layout.positions[t].append((i, j))

    def run(self):
        self.population = self.initialize_population()
        generation = 0
        try:
            while True:
                generation += 1
                fitnesses = [self.simulate(ind) for ind in self.population]
                paired = list(zip(self.population, fitnesses))
                paired.sort(key=lambda x: x[1], reverse=True)
                elites = [p for p,_ in paired[:self.top_k]]
                best_fit = paired[0][1]
                print(f"Gen {generation} best={best_fit}", flush=True)
                if best_fit > self.best_fitness:
                    self.best_fitness = best_fit
                    self.best_layout = paired[0][0]
                newpop = elites.copy()
                while len(newpop) < self.pop_size:
                    p1, p2 = self.select_parents([f for _,f in paired])
                    child = self.crossover(p1, p2)
                    self.mutate(child)
                    newpop.append(child)
                self.population = newpop
        except KeyboardInterrupt:
            print("\nInterrupted!", flush=True)
            print(f"Best fitness={self.best_fitness}")
            self.best_layout.print(self.rows, self.cols)
            data = []
            for ind, fit in paired[:self.top_k]:
                data.append({'fitness': fit, 'layout': ind.to_dict()})
            fn = 'best_layouts.json'
            with open(fn, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved top {self.top_k} layouts to {fn}")

if __name__=='__main__':
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    opt = GeneticOptimizer(path)
    opt.run()
