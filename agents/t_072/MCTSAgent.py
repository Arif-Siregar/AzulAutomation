# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Alec Miller
# Date:    04/01/2021
# Purpose:  Implements an monte carlo tree search agent for the COMP90054 competitive game environment.
#           This is different to mcts_clean.py, which uses mcts_utils_orig.py instead of mcts_utils_clean.py.
#           For yet to be known reasons, MCTSAgent.py has better performance, despite little difference in design.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


from Azul.azul_model import AzulGameRule as GameRule
from .mcts_utils_orig import MCTS
from .mab import UpperConfidenceBounds as UCB
from .qfunc import QTable

THINKTIME   = 0.9
NUM_PLAYERS = 2


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


# Defines this agent.
class myAgent():
    def __init__(self, _id):
        self.id = _id # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS)

    # Generates actions from this state.
    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

    # Take a list of actions and an initial state, and perform Monte Carlo tree search within a time limit.
    # Return the action with the greatest q-value
    def SelectAction(self, actions, rootstate):
        if len(actions) == 1:
            return actions[0]
        qfunction = QTable()
        opp_qfunction = QTable()
        bandit = UCB()
        monte_carlo_ts = MCTS(self.game_rule, rootstate, self.id, qfunction, opp_qfunction, bandit)
        mc_rootnode = monte_carlo_ts.mcts()
        return mc_rootnode.get_value()[1]
        
    
# END FILE -----------------------------------------------------------------------------------------------------------#