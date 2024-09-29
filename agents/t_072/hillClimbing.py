import time
import Azul.azul_utils as utils
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque

THINKTIME   = 0.9
NUM_PLAYERS = 2

class myAgent():
    def __init__(self, _id):
        self.id = _id # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS) # Agent stores an instance of GameRule, from which to obtain functions.
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.

    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)
    
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        score_to_beat = 0
        new_actions = self.GetActions(rootstate)
        best_action = None

        i = 0
        while (i < len(new_actions)) and (time.time()-start_time < THINKTIME):
            temp_old_state = deepcopy(rootstate)
            temp_new_state = self.game_rule.generateSuccessor(temp_old_state, new_actions[i], self.id)
            new_round_score , _ = temp_new_state.agents[self.id].ScoreRound()
            new_end_game_score = temp_new_state.agents[self.id].EndOfGameScore()
            temp_total = new_round_score + new_end_game_score
            if temp_total >= score_to_beat:
                score_to_beat = temp_total
                best_action = new_actions[i]
            i += 1

        return best_action
