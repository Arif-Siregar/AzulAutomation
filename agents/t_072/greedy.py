# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Alec Miller
# Date:    04/05/2023
# Purpose: Implements a greedy player for Azul.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time, random, math, traceback
import Azul.azul_utils as utils
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque

THINKTIME   = 0.95
NUM_PLAYERS = 2


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


# Defines this agent.
class myAgent():
    def __init__(self, _id):
        self.id = _id # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS) # Agent stores an instance of GameRule, from which to obtain functions.
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.
        self.opp_id = (self.id + 1) % 2

    # Generates actions from this state.
    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)
    
    # Calculates a reward for an action in a particular state based on the end of round and end of game scores that
    # would be given
    def RewardAction(self, state, action):
        reward = 0
        state = self.game_rule.generateSuccessor(state, action, self.id)
        next_round_score, _ = state.agents[self.id].ScoreRound()
        end_game_score = state.agents[self.id].EndOfGameScore()
        reward = next_round_score + end_game_score
        return reward

    # Take a list of actions and an initial state, and select the action with greatest reward
    def SelectAction(self, actions, rootstate):
        state = deepcopy(rootstate) # Get the current state
        actions = self.GetActions(state) # Obtain new actions available to the agent in this state.
        max_reward = -math.inf
        best_action = None
        for a in actions: # Then, for each of these actions...
            next_state = deepcopy(state)
            reward = self.RewardAction(next_state, a) # Check for reward of this action
            if reward > max_reward:
                max_reward = reward
                best_action = a
        return best_action

 


    




# END FILE -----------------------------------------------------------------------------------------------------------#