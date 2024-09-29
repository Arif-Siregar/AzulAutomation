# Classes for building a Tree for Monte Carlo Tree Search (MCTS)
# Largely from class notes: https://gibberblot.github.io/rl-notes/single-agent/mcts.html

# This is very similar to mcts_utils_orig.py, but with some modifications and code cleanup
# mcts_clean.py, which uses mcts_utils_clean.py for reasons not ye know has decreased performance
# compared to MCTSAgent, which uses mcts_utils_orig

import time
import random
from collections import defaultdict
from copy import deepcopy
import traceback
import numpy as np


class MCTS:
    def __init__(self, game_rule, rootstate, player_id, qfunction, opp_qfunction, bandit):
        self.game_rule = game_rule
        self.rootstate = rootstate
        self.player_id = player_id
        self.qfunction = qfunction
        self.opp_qfunction = opp_qfunction
        self.bandit = bandit

    """
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    """

    def mcts(self, timeout=0.9, root_node=None):
        if root_node is None:
            root_node = self.create_root_node()

        try:
            start_time = time.time()
            current_time = time.time()
            while current_time < start_time + timeout:
                selected_node = root_node.select()
                child = selected_node
                if selected_node.state.TilesRemaining():
                    child = selected_node.expand()

                rewards = self.simulate(child)
                selected_node.back_propagate(rewards, child)

                current_time = time.time()
            return root_node
        
        except:
            traceback.print_exc()
        

    """ Create a root node representing an initial state """
    def create_root_node(self):
        return Node(
            self.game_rule, None, self.rootstate, self.qfunction, self.opp_qfunction, self.bandit, self.player_id, self.player_id
        )


    """ Choose a random action. Heuristics can be used here to improve simulations. """

    def choose(self, state, player):
        return random.choice(reduce_actions(self.game_rule.getLegalActions(state, player)))

    """ Simulate until a terminal state """

    def simulate(self, node):
        state = deepcopy(node.state)
        cumulative_reward = 0.0
        cumulative_opp_reward = 0.0
        depth = 0
        discount_factor = 0.9
        player = node.player_id
        while self.game_rule.getLegalActions(state, player)[0] != "ENDROUND":
            # Choose an action to execute
            action = self.choose(state, player)

            # Execute the action
            next_state = deepcopy(state)
            next_state = self.game_rule.generateSuccessor(state, action, player)

            # Discount the reward
            reward, opp_reward = get_rewards(state, next_state, self.player_id)
            cumulative_reward += pow(discount_factor, depth) * reward
            cumulative_opp_reward += pow(discount_factor, depth) * opp_reward
            depth += 1

            state = next_state
            player = 1 - player

        return cumulative_reward, cumulative_opp_reward, depth


class Node:
    # Record a unique node id to distinguish duplicate states
    next_node_id = 0

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(
        self,
        game_rule,
        parent,
        state,
        qfunction,
        opp_qfunction,
        bandit,
        root_id,
        player_id,
        reward=0.0,
        opp_reward=0.0,
        action=None,
    ):
        self.game_rule = game_rule
        self.parent = parent
        self.state = state
        self.player_id = player_id
        self.root_id = root_id
        self.node_id = Node.next_node_id
        Node.next_node_id += 1

        # The Q functions used to store state-action values
        self.qfunction = qfunction
        self.opp_qfunction = opp_qfunction

        # A multi-armed bandit for this node
        self.bandit = bandit

        # The immediate rewards received for reaching this state, used for backpropagation
        self.reward = reward # TODO: figure out intermediate rewards
        self.opp_reward = opp_reward

        # The action that generated this node
        self.action = action

        # actions to choose from
        self.actions = reduce_actions(self.game_rule.getLegalActions(self.state, self.player_id))
        # A dictionary from actions to a set of nodes
        self.children = {}

    """ Return true if and only if all child actions have been expanded """

    def get_value(self):
        max_q_value, best_action = self.qfunction.get_max_q(
            self.state, self.actions)
        #max_q_value, best_action = self.qfunction.get_max_q_diff(
            #self.state, self.actions, self.opp_qfunction)
        return max_q_value, best_action

    """ Get the number of visits to this state """

    def get_visits(self):
        return Node.visits[self.state]
    
    
    """ Checks if a node has been fully expanded""" # TODO: Change so node expanded all at once

    def is_fully_expanded(self):
        return len(self.actions) == len(self.children)

    """ Select a node that is not fully expanded """

    def select(self):
        if not self.is_fully_expanded() or not self.state.TilesRemaining():
            return self
        else:
            actions = list(self.children.keys())
            if self.player_id == self.root_id:
                action = self.bandit.select(self.state, actions, self.qfunction)
            else:
                action = self.bandit.select(self.state, actions, self.opp_qfunction)
            return self.get_outcome_child(action).select()

    """ Expand a node if it is not a terminal node """

    def expand(self):
        if self.state.TilesRemaining():
            # Randomly select an unexpanded action to expand
            actions = self.actions - self.children.keys()
            action = random.choice(list(actions))
            return self.get_outcome_child(action)
        return self

    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, rewards, child):
        reward, opp_reward, depth = rewards #TODO: remove depth from rewards, we don't need it here
        discount_factor = 0.9
        action = child.action

        Node.visits[self.state] += 1
        Node.visits[(self.state, action)] += 1
        q_value = self.qfunction.get_q_value(self.state, action)
        opp_q_value = self.opp_qfunction.get_q_value(self.state, action)
        reward = child.reward + reward * discount_factor
        opp_reward = child.opp_reward + opp_reward * discount_factor
        delta = (1 / (Node.visits[(self.state, action)])) * (reward - q_value)
        opp_delta = (1 / (Node.visits[(self.state, action)])) * (opp_reward - opp_q_value)
        self.qfunction.update(self.state, action, delta)
        self.opp_qfunction.update(self.state, action, opp_delta)
        if self.parent != None:
            #self.parent.back_propagate((reward * discount_factor, opp_reward * discount_factor, depth-1), self)
            self.parent.back_propagate((reward, opp_reward, depth), self)


    def get_outcome_child(self, action):
        if action in self.children:
            return self.children[action]
        next_state = deepcopy(self.state)
        #reward = next_state.agents[self.root_id].ScoreRound()[0] + next_state.agents[self.root_id].EndOfGameScore() - self.reward
        #opp_reward = next_state.agents[1-self.root_id].ScoreRound()[0] + next_state.agents[1-self.root_id].EndOfGameScore() - self.opp_reward
        next_state = self.game_rule.generateSuccessor(next_state, action, self.player_id)
        reward, opp_reward = get_rewards(self.state, next_state, self.root_id)
        new_child = Node(
            self.game_rule, self, next_state, self.qfunction, self.opp_qfunction, self.bandit, self.root_id, 1-self.player_id, reward, opp_reward, action
        )

        self.children[action] = new_child
        return new_child


""" Create a list of actions for MCTS to consider, removing the clearly bad ones to save time """

def reduce_actions(actions):
    reduced_actions = []
    a = 20
    while len(reduced_actions) == 0:
        for action in actions:
            #if len(actions) > 2*a and action[2].num_to_floor_line != 0:
                #continue
            if len(actions) > a and action[2].num_to_pattern_line == 0:
                continue
            reduced_actions.append(action)
        a *= 2
    return reduced_actions


""" Get the rewards achieved from taking an action"""

def get_rewards(prev_state, state, id):
    prev_score = get_score(prev_state, id)
    prev_opp_score = get_score(prev_state, 1-id)
    score = get_score(state, id)
    opp_score = get_score(state, 1-id)

    return score - prev_score, opp_score - prev_opp_score

def get_score(state, id):
    return state.agents[id].ScoreRound()[0] + state.agents[id].EndOfGameScore() - get_penalties(state, id) #+ get_bonuses(state, id) #

def get_bonuses(state, id):
    return columns_bonus(state, id) + first_player_bonus(state, id) + colour_bonus(state, id) + middle_bonus(state, id)

def get_penalties(state, id):
    return penalize_unfinished_line(state, id)

def first_player_bonus(state, id):
    return int(state.next_first_agent==id and (max(
        sum(state.agents[1-id].grid_state[0]), sum(state.agents[id].grid_state[0]))<5))

def columns_bonus(state, id):
    grid_state = np.asarray(state.agents[id].grid_state)
    bonus = 0
    
    for i in range(len(grid_state)):
        col_sum = sum(grid_state[:,i])
        if col_sum == 3:
            bonus += 1
        elif col_sum == 4:
            bonus += 3
    
    return bonus# * multiplier

def colour_bonus(state, id):
    grid_state = np.asarray(state.agents[id].grid_state)
    bonus = 0
    #if sum(grid_state[0])==5:
    #    return bonus
    print(grid_state)
    diagonals = [np.diagonal(np.roll(grid_state, i, axis=1)) for i in range(5)]
    for diagonal in diagonals:
        diag_sum = sum(diagonal)
        if diag_sum == 3:
            bonus += 1
        elif diag_sum == 4:
            bonus += 3
        
    return bonus

def middle_bonus(state, id):
    grid_state = np.asarray(state.agents[id].grid_state)
    multiplier = max(0,(2-max(sum(state.agents[1-id].grid_state[0]), sum(state.agents[id].grid_state[0]))))
    print(multiplier * (2*sum(grid_state[:,3]) + sum(grid_state[:,2]) + sum(grid_state[:,4])))
    return multiplier * (2*sum(grid_state[:,3]) + sum(grid_state[:,2]) + sum(grid_state[:,4]))

def penalize_unfinished_line(state, id):
    lines_number = np.asarray(state.agents[id].lines_number)
    penalty = 0

    for i in range(len(lines_number)):
        tiles_on_line = lines_number[i]
        if tiles_on_line > 0 and tiles_on_line < i+1:
            penalty += (0.5*(i + 1 - tiles_on_line))

    return penalty
    