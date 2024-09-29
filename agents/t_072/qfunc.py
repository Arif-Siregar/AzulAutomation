# q-function for MCTS and MAB algorithms

from collections import defaultdict
from abc import abstractmethod
from math import inf

class QFunction:

    def __init__(self):
        pass

    @abstractmethod
    def get_q_value(self, state, action):
        pass

    @abstractmethod
    def get_max_q(self, state, actions):
        pass

    @abstractmethod
    def get_max_q_diff(self, state, actions, qtable2):
        pass

    @abstractmethod
    def update(self, state, action, delta):
        pass


class QTable(QFunction):
    def __init__(self, default=0.0):
        self.qtable = defaultdict(lambda: default)

    def get_q_value(self, state, action):
        return self.qtable[(state, action)]
    
    def get_max_q(self, state, actions):
        max_q = -inf
        best_action = None
        for action in actions:
            q = self.qtable[(state, action)]
            if q > max_q:
                max_q = q
                best_action = action
        return max_q, best_action
    
    def get_max_q_diff(self, state, actions, qtable2):
        max_diff = -inf
        best_action = None
        for action in actions:
            q1 = self.qtable[(state, action)]
            q2 = qtable2.qtable[(state, action)]
            diff = q1 - q2
            if diff > max_diff:
                max_diff = diff
                best_action = action
        return max_diff, best_action

    def update(self, state, action, delta):
        self.qtable[(state, action)] = self.qtable[(state, action)] + delta