# Azul Game Agent implementing DeepQ method
import importlib
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState as InitalState
import Azul.azul_utils as utils
from template import Agent as DummyAgent
from alec import myAgent as AlecAgent
from game import Game
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from func_timeout import func_timeout, FunctionTimedOut
import random, time, copy

import logging

THINK_TIME=0.95
NUM_PLAYERS=2

# Constants for training
IN_MODEL_PATH = "models/DQNModelv5_3.h5"
OUT_MODEL_PATH = "models/DQNModelv5_4.h5"
ACTION_PATH = "observed_action.json"

# Constants for running General game runner
# MODEL_PATH = "agents/t_072/DQNModel.h5"
# ACTION_PATH = "agents/t_072/observed_action.json"

# Plays a game against the random agent
# Referenced from game.py
class AdvancedGame(Game):
    def __init__(self, GameRule,
                 agent_list, 
                 num_of_agent,
                 max_iter_ep,
                 agents_namelist = ["Alice","Bob"]):
        super().__init__(GameRule,
                 agent_list, 
                 num_of_agent,
                 displayer = None, 
                 agents_namelist = agents_namelist)
        self.max_iter_ep = max_iter_ep
        
    def Run(self):
        history = {"actions":[]}
        action_counter = 0
        while not self.game_rule.gameEnds() and action_counter <= self.max_iter_ep:
            agent_index = self.game_rule.getCurrentAgentIndex()
            agent = self.agents[agent_index] if agent_index < len(self.agents) else self.gamemaster
            game_state = self.game_rule.current_game_state
            game_state.agent_to_move = agent_index
            actions = self.game_rule.getLegalActions(game_state, agent_index)
            actions_copy = copy.deepcopy(actions)
            gs_copy = copy.deepcopy(game_state)

            # Delete all specified attributes in the agent state copies, if this isn't a perfect information game.
            if self.game_rule.private_information:
                delattr(gs_copy.deck, 'cards') # Upcoming cards cannot be observed.
                for i in range(len(gs_copy.agents)):
                    if gs_copy.agents[i].id != agent_index:
                        for attr in self.game_rule.private_information:
                            delattr(gs_copy.agents[i], attr)
            # Freedom is given to agents, let them return any action in any time period, at the risk of breaking 
            #the simulation.
            selected = agent.SelectAction(actions_copy, gs_copy) 
            
            # For DQN Agent, store experince replay (set DQNagent ID to 1)
            if agent_index < len(self.agents):
                # Store player agent er
                next_state, reward, done = self.step(gs_copy, selected, agent_index)
                self.agents[1].StoreExperience(agent_index, gs_copy, selected, reward, next_state, done)
            random.seed(self.seed_list[self.seed_idx])
            self.seed_idx += 1
            history["actions"].append({action_counter:{"agent_id":self.game_rule.current_agent_index,"action":selected}})
            action_counter += 1
            
            self.game_rule.update(selected)
            random.seed(self.seed_list[self.seed_idx])
            self.seed_idx += 1

            if (agent_index != self.game_rule.num_of_agent) and (self.warnings[agent_index] == self.warning_limit):
                history = self._EndGame(self.game_rule.num_of_agent,history,isTimeOut=True,id=agent_index)
                return history
                
        # Score agent bonuses
        return self._EndGame(self.game_rule.num_of_agent,history,isTimeOut=False)
    
    def step(self, current_state, action, a_id):
        done=False
        gs_copy = copy.deepcopy(current_state)
        gs_copy_2 = copy.deepcopy(gs_copy) # Becomes useless after line 104
        gs_copy_3 = copy.deepcopy(gs_copy)
        gs_copy_4 = copy.deepcopy(gs_copy)
        ns = self.game_rule.generateSuccessor(gs_copy_2,action,a_id)
        ns_copy = copy.deepcopy(ns)
        ns_copy_2 = copy.deepcopy(ns)
        ns_copy_3 = copy.deepcopy(ns)
        
        # Calculate rewards
        reward = 0

        # Current game score
        curr_round_score, _ = gs_copy_3.agents[a_id].ScoreRound()
        curr_bonus = gs_copy_3.agents[a_id].EndOfGameScore()        

        next_round_score, _ = ns_copy_2.agents[a_id].ScoreRound()
        next_bonus =  ns_copy_2.agents[a_id].EndOfGameScore()

        additional_score = next_round_score - curr_round_score
        additional_bonus = next_bonus - curr_bonus
 
        reward += additional_score + additional_bonus
        
        cur_p_state = gs_copy_4.agents[a_id]
        next_p_state = ns_copy_3.agents[a_id]

        # Intermediate rewards for choosing and placing tiles into pattern lines
        # these rewards are weighed less than score rewards to encourage high scores
        for row in range(cur_p_state.GRID_SIZE):
            # Add reward for additional tiles to pattern lines / row number
            reward += (next_p_state.lines_number[row] - cur_p_state.lines_number[row]) / (row+1)

        if self.game_rule.gameEnds():
            done=True
        return ns, reward, done


class myAgent():
    def __init__(self, _id, training=False, model_path=IN_MODEL_PATH):
        self.id = _id
        self.game_rule = GameRule(NUM_PLAYERS)
        self.game_info = GameInfo()
        self.state_size = self.game_info.getStateSize()

        self.training=training
        self.model_path = model_path

        self.learning_rate = 0.0025
        self.decay = 0.90
        self.epsilon = 0.1
        self.min_epsilon = 0.1
        self.eplison_decay = 0.005
        self.batch_size = 32

        # Number of ER experienced by Agent
        self.exp_count = 0

        # Number of steps before updating target model
        self.train_steps = 0
        self.update_target_model = 500

        # Initialize memory buffer for experience replay
        self.replay_memory = list()
        self.max_memory = 1000

        self.observed_actions = self.load_actions()
        self.max_action_size = self.game_info.getActionSize()

        # Create NN model for Q-function
        self.model = create_model(self.state_size, self.max_action_size)
        if not IN_MODEL_PATH == "": 
            self.model.load_weights(IN_MODEL_PATH)

        if self.training:            
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(self.learning_rate))
            self.target_model = create_model(self.state_size, self.max_action_size)
            # Target model (Starts off with same weights as original model)
            self.target_model.set_weights(self.model.get_weights())
            self.target_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(self.learning_rate))
        

    # Stores a new ER into the replay memory of the model
    def StoreExperience(self, agent_id, current_state, action ,reward, next_state, done):
        self.replay_memory.append({
            "agent_id": agent_id,
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done":done
        })
        if len(self.replay_memory) > self.max_memory:
            self.replay_memory.pop(0)
        self.exp_count += 1
    
    # Checks whether the Agent has enough experience to train the model
    def hasEnoughER(self):
        return self.exp_count >= self.batch_size

    # Decrease epsilon by epsilon_decay rate
    def decayEpsilon(self):
        self.epsilon *= np.exp(-self.eplison_decay)
        self.epsilon = max(self.min_epsilon, self.epsilon) 

    # Returns an action for the DQNAgent
    def SelectAction(self, actions, cur_state):
        # Exploration (only in training)
        if self.training:
            if random.random() < self.epsilon:
                random_action = random.choice(actions)
                self.actionToInt(random_action)
                return random_action
        # Exploitation (default in production)
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

    # Train model with stored experience
    def train(self):
        random.shuffle(self.replay_memory)
        batch = self.replay_memory[:self.batch_size]

        for exp in batch:
            agent_id = exp["agent_id"]
            state_feature = getPlayerState(exp["current_state"].agents[agent_id], exp["current_state"])
            q_values = self.model.predict(state_feature)
            target_q = exp["reward"]
            next_legal_actions = self.game_rule.getLegalActions(exp["next_state"], agent_id)
            if 'ENDROUND' in next_legal_actions or 'STARTROUND' in next_legal_actions:
                continue
            next_actions_ints = [self.actionToInt(action) for action in next_legal_actions]
            # If not terminal state, update reward
            if not exp["done"]:
                next_q_values = self.target_model.predict(getPlayerState(exp["next_state"].agents[agent_id], exp["next_state"]))[0]
                max_q = 0
                for i in next_actions_ints:
                    max_q = max(next_q_values[i], max_q)
                # Update the q-value for 
                target_q = target_q + self.decay * max_q
            q_values[0][self.actionToInt(exp["action"])] = target_q
            # Need to fit with the new output vector!!!
            self.model.fit(state_feature, np.array([q_values[0]]))
            self.train_steps += 1
        
        # copy model weights to target q model
        if self.train_steps % self.update_target_model == 0:
            self.target_model.set_weights(self.model.get_weights())

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

    # Save/overwrite observed actions to file (json format)
    def save_actions(self):
        with open(ACTION_PATH, 'w') as f:
            json.dump(self.observed_actions, f)

    # Saves model weights. (default H5 format)
    def save_model(self, path=OUT_MODEL_PATH):
        self.model.save(path)

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
    return Sequential([
        Dense(units=512, input_dim=state_size, activation='relu'),
        Dense(units=256, activation='relu'),
        Dense(units=action_size, activation='linear')
    ])

def getLogger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log


## Train the model
if __name__ == "__main__":
    log = getLogger("dqn-trainer")
    log.info("Starting program...")
    # Number of episodes to run
    n_episodes = 300
    # Max number of iterations per episode
    max_iter_ep = 100

    # game_rule = GameRule(NUM_PLAYERS)
    # import GameRule
    model = importlib.import_module(f"Azul.azul_model")
    _GameRule = getattr(model, f'AzulGameRule')

    # Initalise DQNAgent and RandomAgent to play against each other
    num_agents = NUM_PLAYERS
    # random_agent = DummyAgent(0)
    alec_agent = AlecAgent(0)
    # Initialises model to train
    dqn_agent = myAgent(1, training=True)
    agent_list = []
    agent_list.append(alec_agent)
    agent_list.append(dqn_agent)

    agents_namelist = ["random","dqn"]

    # Keep count of number of wins for each agent (0:Random Agent, 1:DQNAgent)
    win_count = {0:0, 1:0}
    avg_score = {0:0, 1:0}

    # Run game for n_episode times
    for e in range(n_episodes):
        # Initialise new game
        game = AdvancedGame(_GameRule,
                            agent_list,
                            num_agents,
                            max_iter_ep,
                            agents_namelist=agents_namelist)
        
        log.info(f"Running game {e+1}")
        log.info(f"Epsilon: {dqn_agent.epsilon}")
        # Run episode
        history = game.Run()

        # # Decay epsilon
        # dqn_agent.decayEpsilon()
        # RandomAgent wins (May be running score)
        txt=""
        if history["scores"][0] > history["scores"][1]:
            txt = "Random agent wins!"
            win_count[0] += 1 
        # DQNAgent wins
        elif history["scores"][0] < history["scores"][1]:
            txt = "DQN agent wins!"
            win_count[1] += 1
        else:
            txt = "Draw!"
        avg_score[0] = round((avg_score[0]*e + history["scores"][0])/(e+1),2)
        avg_score[1] = round((avg_score[1]*e + history["scores"][1])/(e+1),2)

        log.info(f"{txt} Scores: {history['scores'][0]} - {history['scores'][1]}")
        log.info(f"Win count - random:{win_count[0]}, dqn:{win_count[1]}")
        log.info(f"Average score - random:{avg_score[0]}, dqn:{avg_score[1]}")

        # Train dqn agent if it has enough experiences
        if dqn_agent.hasEnoughER():
            log.info("Training model...")
            dqn_agent.train()
    
    # Save observed actions
    dqn_agent.save_actions()
    # Store the model after training:
    log.info("Saving model...")
    dqn_agent.save_model()
    log.info("Model save complete!")
    log.info("Terminating execution...")
        

        
        



