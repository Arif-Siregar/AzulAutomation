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


"""
Class executes the MCTS algorithm and builds an MCTS tree
"""
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

                # Selection step
                selected_node = root_node.select()
                child = selected_node

                # Expansion step
                if selected_node.state.TilesRemaining():
                    child = selected_node.expand()

                # Simulation step
                rewards = self.simulate(child)

                # Backpropagation step
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

            # Discount the rewards
            reward, opp_reward = get_rewards(state, next_state, self.player_id)
            cumulative_reward += pow(discount_factor, depth) * reward
            cumulative_opp_reward += pow(discount_factor, depth) * opp_reward
            depth += 1

            # Continue from the next state as the opposition if the end of round isn't reached
            state = next_state
            player = 1 - player

        return cumulative_reward, cumulative_opp_reward


"""
Node for a MCTS tree
"""
class Node:

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

        # The Q functions used to store state-action values
        self.qfunction = qfunction
        self.opp_qfunction = opp_qfunction

        # A multi-armed bandit for this node
        self.bandit = bandit

        # The immediate rewards received for reaching this state, used for backpropagation
        self.reward = reward
        self.opp_reward = opp_reward

        # The action that generated this node
        self.action = action

        # actions to choose from
        self.actions = reduce_actions(self.game_rule.getLegalActions(self.state, self.player_id))
        # A dictionary from actions to a set of nodes
        self.children = {}


    """ Return the best action found by the MCTS algorithm """

    def get_best_action(self):
        _, best_action = self.qfunction.get_max_q(
            self.state, self.actions)
        return best_action


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
            # if action is fully expanded, select the best action using MAB
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
        reward, opp_reward = rewards
        discount_factor = 0.9
        action = child.action

        Node.visits[self.state] += 1

        # get current q-values
        q_value = self.qfunction.get_q_value(self.state, action)
        opp_q_value = self.opp_qfunction.get_q_value(self.state, action)

        # backpropagate reward from simulation
        reward = child.reward + reward * discount_factor
        opp_reward = child.opp_reward + opp_reward * discount_factor

        # update q-values
        delta = (1 / (self.get_visits())) * (reward - q_value)
        opp_delta = (1 / (self.get_visits())) * (opp_reward - opp_q_value)
        self.qfunction.update(self.state, action, delta)
        self.opp_qfunction.update(self.state, action, opp_delta)

        # continue backpropagation to root node
        if self.parent != None:
            self.parent.back_propagate((reward, opp_reward), self)


    """ Get the child node reached from the given action """
    def get_outcome_child(self, action):

        # get the node if it already exists
        if action in self.children:
            return self.children[action]
        
        # otherwise execute the action and create a new child node
        next_state = deepcopy(self.state)
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
    a = 10
    while len(reduced_actions) == 0:
        for action in actions:
            # filter out actions where all tiles are placed on the pattern line
            if len(actions) > a and action[2].num_to_pattern_line == 0:
                continue
            reduced_actions.append(action)
        a *= 2
    return reduced_actions


""" Get the rewards achieved from taking an action """

def get_rewards(prev_state, state, id):
    prev_score = get_score(prev_state, id)
    prev_opp_score = get_score(prev_state, 1-id)
    score = get_score(state, id)
    opp_score = get_score(state, 1-id)

    return score - prev_score, opp_score - prev_opp_score


""" Calculate the score of an agent if the game ended in this state, as well strategic bonuses and penalties """

def get_score(state, id):
    return state.agents[id].ScoreRound()[0] + state.agents[id].EndOfGameScore() - get_penalties(state, id)# + get_bonuses(state, id) 


""" Calculate strategic bonuses for moves that may help earn points in later rounds """

def get_bonuses(state, id):
    return columns_bonus(state, id) + first_player_bonus(state, id) + colour_bonus(state, id) + middle_bonus(state, id)


""" Calculate the score of an agent if the game ended in this state, as well strategic bonuses and penalties """

def get_penalties(state, id):
    return penalize_unfinished_line(state, id)


""" Calculate the score of an agent if the game ended in this state, as well strategic bonuses and penalties """

def first_player_bonus(state, id):
    return 2*int(state.next_first_agent==id and not game_ending(state))


""" Calculate a bonus for having multiple tiles in a column """

def columns_bonus(state, id):

    # don't keep giving rewards game is in final round
    # if game_ending(state):
    #    return 0

    grid_state = np.asarray(state.agents[id].grid_state)
    bonus = 0

    # multiplier that gives greater rewards in earlier rounds, which has proved ineffective
    #multiplier = max(0,(5-max(sum(state.agents[1-id].grid_state[0]), sum(state.agents[id].grid_state[0]))))

    for i in range(len(grid_state)):
        col_sum = sum(grid_state[:,i])
        # small bonus for 3 in a column
        if col_sum == 3:
            bonus += 1
        # larger bonus for 4 in a column
        elif col_sum == 4:
            bonus += 2
    
    return bonus# * multiplier


""" Calculate a bonus for having multiple tiles of a colour. Shown to be ineffective """

def colour_bonus(state, id):

    #if game_ending(state):
    #    return 0

    grid_state = np.asarray(state.agents[id].grid_state)
    bonus = 0

    # multiplier that gives greater rewards in earlier rounds, which has proved ineffective
    #multiplier = max(0,(2-max(sum(state.agents[1-id].grid_state[0]), sum(state.agents[id].grid_state[0]))))

    diagonals = [np.diagonal(np.roll(grid_state, i, axis=1)) for i in range(5)]
    for diagonal in diagonals:
        diag_sum = sum(diagonal)
        if diag_sum == 3:
            bonus += 1
        elif diag_sum >= 4:
            bonus += 2
        
    return bonus# * multiplier


""" Calculate a bonus for building from the middle columns early in the game. Shown to be ineffective """

def middle_bonus(state, id):
    grid_state = np.asarray(state.agents[id].grid_state)

    # multiplier to greater reward building from the middle earlier
    multiplier = max(0,(2-max(sum(state.agents[1-id].grid_state[0]), sum(state.agents[id].grid_state[0]))))

    # return greater bonus the closer to the middle the tiles are
    return multiplier * (2*sum(grid_state[:,3]) + sum(grid_state[:,2]) + sum(grid_state[:,4]))


""" Penalise leaving tile lines unfinished """

def penalize_unfinished_line(state, id):
    lines_number = np.asarray(state.agents[id].lines_number)
    penalty = 0

    # constant penalty for any started but unfinished line, plus extra penalties for multiple unfilled slots
    for i in range(len(lines_number)):
        tiles_on_line = lines_number[i]
        if tiles_on_line > 0 and tiles_on_line < i+1:
            penalty += 0.5*(i + 1 - tiles_on_line)

    return penalty
    

""" Returns true if the top row is filled for any player, meaning the game is ending this round """

def game_ending(state):
    return max(sum(state.agents[0].grid_state[0]), sum(state.agents[1].grid_state[0]))<5