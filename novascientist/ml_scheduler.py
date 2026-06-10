import json
import math
import random
import os
from collections import defaultdict
from typing import Tuple

class MultiArmedBanditScheduler:
    """
    An ML-based adaptive scheduler using the Upper Confidence Bound (UCB1) algorithm.
    It dynamically selects the most promising generation strategies (mode, reasoning_type)
    based on historical success rates (ELO rankings of generated hypotheses).
    """
    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        self.history_file = os.path.join(state_dir, "bandit_history.json")
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.total_pulls = 0
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.counts = defaultdict(int, data.get('counts', {}))
                    self.values = defaultdict(float, data.get('values', {}))
                    self.total_pulls = data.get('total_pulls', 0)
            except Exception:
                pass

    def _save_history(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump({
                'counts': dict(self.counts),
                'values': dict(self.values),
                'total_pulls': self.total_pulls
            }, f)

    def select_arm(self, arms: list[str]) -> str:
        """
        Selects an arm using UCB1 algorithm.
        """
        # Always explore unplayed arms first
        for arm in arms:
            if self.counts[arm] == 0:
                return arm
        
        # Calculate UCB values
        ucb_values = {}
        for arm in arms:
            exploitation = self.values[arm]
            # Exploration factor (c=1.414 is standard for UCB1)
            exploration = 1.414 * math.sqrt(math.log(self.total_pulls) / self.counts[arm])
            ucb_values[arm] = exploitation + exploration
            
        return max(ucb_values, key=ucb_values.get)

    def update_arm(self, arm: str, reward: float):
        """
        Updates the value of an arm based on observed reward (e.g., ELO score improvement).
        """
        self.counts[arm] += 1
        self.total_pulls += 1
        
        n = self.counts[arm]
        value = self.values[arm]
        # Incremental mean update
        new_value = ((n - 1) * value + reward) / n
        self.values[arm] = new_value
        
        self._save_history()

    def get_strategy(self, modes: list[str], reasoning_types: list[str]) -> Tuple[str, str]:
        """
        Returns an optimal (mode, reasoning_type) tuple using the bandit.
        """
        arms = [f"{m}|{rt}" for m in modes for rt in reasoning_types]
        selected_arm = self.select_arm(arms)
        mode, reasoning_type = selected_arm.split("|")
        return mode, reasoning_type
