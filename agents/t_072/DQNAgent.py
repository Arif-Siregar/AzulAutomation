import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState as InitalState
import Azul.azul_utils as utils
from tensorflow import keras
from template import Agent

NUM_PLAYERS=2

MODEL_PATH = "agents/t_072/models/DQNModelv5_4.h5"
ACTION_PATH = "agents/t_072/observed_action.json"

class myAgent(Agent):
    def __init__(self, _id, model_path=MODEL_PATH):
        self.id = _id
        self.game_rule = GameRule(NUM_PLAYERS)
        self.game_info = GameInfo()
        self.state_size = self.game_info.getStateSize()
        self.max_action_size = self.game_info.getActionSize()

        self.model_path = model_path

        self.observed_actions = self.load_actions()

        # Create NN model for Q-function
        self.model = create_model(self.state_size, self.max_action_size) 
        self.model.load_weights(MODEL_PATH)       

    # Returns an action for the DQNAgent
    def SelectAction(self, actions, cur_state):
        legal_actions = []
        for legal_id, action in enumerate(actions):
            action_int = self.actionToInt(action)
            legal_actions.append((action_int,legal_id))
    
        # Q-values returned from model
        q_values = self.model.predict(getPlayerState(cur_state.agents[self.id], cur_state))[0] 
        maxq = 0
        maxidx = 0
        max_action_int = 0
        for action_int,legal_id in legal_actions:
            val = q_values[action_int]
            if val > maxq:
                maxq = val
                maxidx = legal_id
                max_action_int = action_int

        return actions[maxidx]

    # Maps actions to an int
    def actionToInt(self, action):
        tg = action[2]
        actStr = f"{action[1]},{utils.TileToShortString(tg.tile_type)},{tg.pattern_line_dest}"
        # New action observed: add to dictionary
        if actStr not in self.observed_actions:
            self.observed_actions[actStr] = len(self.observed_actions)
        return self.observed_actions[actStr]

    # Load any existing observed actions file (json format)
    def load_actions(self):
        actions = dict()
        if os.path.exists(ACTION_PATH):
            with open(ACTION_PATH, 'r') as f:
                actions = json.load(f)
        return actions


# Helper class to retrieve the specifications of the game to retreive potential
# state and action space size
class GameInfo():
    def __init__(self):
        self.game_state = InitalState(NUM_PLAYERS)
        self.actions = utils.Action
        self.sample_player = self.game_state.agents[0]
        self.grid_size = self.sample_player.GRID_SIZE

    def getStateSize(self):
        size = 0
        # Pattern Lines (5)
        size += sum(range(1,self.grid_size+1))
        # Floor lines (7)
        size += len(self.sample_player.floor)
        # Grid size (25)
        size += self.grid_size**2
        # Num of facotries (+1 for centre pile) * (Tile type + number of tiles in pile)
        size += (len(self.game_state.factories)+1) * 5
    
        return size

    # Returns action size for the game (180 for Azul)
    def getActionSize(self):
        # Number of Factories (5 for 2) + Centre pile (1)
        size = (self.game_state.NUM_FACTORIES[NUM_PLAYERS-2] + 1)
        # Number of tile types
        size *= 5
        # Potential pattern lines to place tiles (5), -1 for all in floor line
        size *= (self.grid_size + 1)

        return size


def create_model(state_size, action_size):
    """
    Creates a NN representing the q-function.
    """
    return keras.models.Sequential([
        keras.layers.Dense(units=512, input_dim=state_size, activation='relu'),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=action_size, activation='linear')
    ])

def getPlayerState(p_state, game_state):
    """
    Returns the current board state in an array
    """

    pattern_line = list()
    grid = list()
    factories = list()

    # Encapture pattern line state
    for row in range(p_state.GRID_SIZE):
        if p_state.lines_tile[row] != -1:
            t_type = p_state.lines_tile[row]
            t_type_str = utils.TileToShortString(t_type)
            t_count = p_state.lines_number[row]

            # Append placed tiles
            for i in range(t_count):
                pattern_line.append(t_type.value)

            # Append 0 for empty spaces
            for i in range(t_count, row+1):
                pattern_line.append(0)
        else:
            # Append 0 for empty rows
            for i in range(row+1):
                pattern_line.append(0)

    # Encapture floor line state (Tile type does not matter here)
    pattern_line.extend(p_state.floor)

    # Encapture player grid state 
    # (-1 if empty, int corresponding to tile type if filled)
    for row in range(p_state.GRID_SIZE):
        for col in range(p_state.GRID_SIZE):
            grid.append(p_state.grid_scheme[row][col] if p_state.grid_state[col][col]==1 else -1)

    pattern_line.extend(grid)

    # Encapture the factories and centre pile
    for td in game_state.factories:
        # For every tile type
        for tile in utils.Tile:
            factories.append(td.tiles[tile])
        
    # Encapture centre piile
    for tile in utils.Tile:
        factories.append(game_state.centre_pool.tiles[tile])

    pattern_line.extend(factories)

    return np.array([pattern_line])