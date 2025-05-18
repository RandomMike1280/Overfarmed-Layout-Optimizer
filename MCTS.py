import numpy as np
import torch
import torch.nn.functional as F
from AlphaFarmer import AlphaFarmer
from farming import FarmGame
from concurrent.futures import ThreadPoolExecutor
from math import sqrt

class Node:
    def __init__(self, game: FarmGame, state: np.ndarray, turn: int, prior: float = 0.0, parent=None):
        self.game = game
        self.state = state
        self.turn = turn
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.multivisit = 0
        self.maxvisit = 0
        self.is_expanded = False
        # Terminal detection: no zeros => terminal
        self.is_terminal = not np.any(self.state == 0)
        self.value = 0.0
        if self.is_terminal:
            val = self.game.get_value_and_terminated(self.state, self.turn)
            self.value = val[0] if isinstance(val, tuple) else val

    def expand(self, policy: np.ndarray):
        valid_moves = self.game.get_valid_moves(self.state)
        probs = policy * valid_moves
        if probs.sum() > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones_like(probs) / len(probs)
        for action, p in enumerate(probs):
            next_state, next_turn = self.game.get_next_state(self.state, action, self.turn)
            self.children[action] = Node(self.game, next_state, next_turn, prior=p, parent=self)
        self.is_expanded = True

    def is_leaf(self):
        return not self.is_expanded

    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

class MCTS:
    def __init__(self, game: FarmGame, model: AlphaFarmer, device: str = 'cpu', cpuct: float = 1.0):
        self.game = game
        self.model = model.to(device)
        self.device = device
        self.cpuct = cpuct
        # Cache NN evaluations for transpositions
        self.nn_cache = {}

    def search(self, state: np.ndarray, num_sims: int, num_threads: int = 1):
        root = Node(self.game, state, turn=0, prior=1.0)
        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(self.execute_one_iteration, root) for _ in range(num_sims)]
                for f in futures:
                    f.result()
        else:
            for _ in range(num_sims):
                self.execute_one_iteration(root)
        return root

    def _backup(self, node: Node, value: float):
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def get_action_probs(self, root: Node, temperature: float = 1.0):
        # Safely compute action probabilities from visit counts
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)
        # Deterministic choice if temperature is zero or too small
        if temperature <= 1e-6:
            best = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[best] = 1.0
        else:
            # Softmax-like of visits^(1/temperature)
            scaled = visits ** (1.0 / temperature)
            total = np.sum(scaled)
            probs = scaled/total if total > 0 else np.ones_like(scaled)/len(scaled)
        return dict(zip(actions, probs))

    def get_best_action(self, root: Node):
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def execute_one_iteration(self, root: Node):
        """Perform one MCTS iteration: select, evaluate (with cache), expand, backup."""
        node = root
        path = []
        # Selection + leaf detection
        while True:
            path.append(node)
            if node.is_terminal:
                value = node.value
                break
            if not node.is_expanded:
                key = node.state.tobytes()
                if key in self.nn_cache:
                    value, policy = self.nn_cache[key]
                else:
                    value, policy = self._evaluate(node)
                    self.nn_cache[key] = (value, policy)
                node.expand(policy)
                break
            # PUCT selection using visits and multivisit
            total = sum(c.visit_count + c.multivisit for c in node.children.values())
            best = None
            best_score = -float('inf')
            for c in node.children.values():
                u = self.cpuct * c.prior * sqrt(total) / (1 + c.visit_count + c.multivisit)
                score = c.q_value() + u
                if score > best_score:
                    best_score, best = score, c
            best.multivisit += 1
            node = best
        # Backup
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            n.maxvisit = max(n.maxvisit, n.visit_count)
            n.multivisit = 0

    def set_root(self, root: Node, action: int) -> Node:
        """After a move, adopt a child as new root or create a fresh one."""
        if action in root.children:
            new_root = root.children[action]
            new_root.parent = None
        else:
            next_state, next_turn = self.game.get_next_state(root.state, action, root.turn)
            new_root = Node(self.game, next_state, next_turn, prior=1.0)
        return new_root

    def _evaluate(self, node: Node):
        encoded = self.game.get_encoded_state(node.state)
        x = torch.from_numpy(encoded).unsqueeze(0).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            value_tensor, policy_tensor = self.model(x)
            value = value_tensor.item()
            policy = F.softmax(policy_tensor, dim=1).cpu().numpy()[0]
        return value, policy

if __name__ == "__main__":
    from farming import FarmGame
    from AlphaFarmer import AlphaFarmer
    from MCTS import MCTS
    grid_size = 3

    game = FarmGame(grid_size)
    model = AlphaFarmer(6, (grid_size, grid_size), game.action_size)
    mcts = MCTS(game, model, device='cpu', cpuct=1.0)

    root = mcts.search(game.get_initial_state(), num_sims=1000, num_threads=4)
    probs = mcts.get_action_probs(root)
    action = mcts.get_best_action(root)

    print(probs)
    print(action)
    state = game.get_initial_state()
    turn = 0
    while turn < 8:
        state, turn = game.get_next_state(state, action, turn)
        print(state)
        root = mcts.set_root(root, action)
        root = mcts.search(state, num_sims=1000, num_threads=4)
        probs = mcts.get_action_probs(root)
        action = mcts.get_best_action(root)
        print(probs)
        print(action)