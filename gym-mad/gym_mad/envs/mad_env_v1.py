import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
import json
import os
from tqdm import tqdm


class MadGameConfig_v1:
    '''
    All game hyperparameters go in this config class
    '''

    def __init__(self, filename=None):
        '''
        This function automatically finds path to config folder
        '''
        if filename is not None:
            self.data = self.load_from_json(filename)
        else:
            self.set_base_attributes()

    def set_base_attributes(self):
        self.data = dict()

        # set common attributes
        for action in MadAction_v1.action_strings:
            self.data[action] = dict()
            self.data[action]["cash_threshold"] = 0
        #    self.data[action]["enemy"] = dict()
        #    self.data[action]["penalty"] = dict()
        #    self.data[action]["player"]["cash_threshold"] = 0
        #    self.data[action]["player"]["military_threshold"] = 0
            # self.data[action]["player"]["requires_threat"] = 0
        #     self.data[action]["player"]["cash_delta"] = 0
        #     self.data[action]["player"]["income_delta"] = 0
        #     self.data[action]["player"]["military_delta"] = 0
        #     self.data[action]["player"]["has_made_threat_set"] = 0
        #     self.data[action]["player"]["has_made_threat_clear"] = 0
        #     self.data[action]["enemy"]["cash_delta"] = 0
        #     self.data[action]["enemy"]["income_delta"] = 0
        #     self.data[action]["enemy"]["military_delta"] = 0
        #     self.data[action]["penalty"]["insufficient_cash"] = 0
        #     self.data[action]["penalty"]["insufficient_military"] = 0
        #     self.data[action]["penalty"]["insufficient_threat"] = 0
        #     self.data[action]["reward"] = 0

        action_invest_eco = MadAction_v1.action_invest_economy
        action_invest_mil = MadAction_v1.action_invest_military
        action_attack = MadAction_v1.action_attack
        action_threaten = MadAction_v1.action_threaten
        action_nuke = MadAction_v1.action_nuke

        # set action-specific attributes
        action_invest_eco_fields = ["income_delta", "cash_delta", "log_coefficient", "reward_offset", "min_reward"]
        for field in action_invest_eco_fields:
            self.data[action_invest_eco][field] = 0

        action_invest_mil_fields = ["cash_delta", "military_delta", "log_coefficient", "military_cash_scale_factor"]
        for field in action_invest_mil_fields:
            self.data[action_invest_mil][field] = 0

        action_atk_fields = ["L_cash", "L_miltary", "log_coefficient", "log_epsilon", "military_threshold"]
        for field in action_atk_fields:
            self.data[action_attack][field] = 0

        action_threaten_fields = ["reward", "military_threshold"]
        for field in action_threaten_fields:
            self.data[action_threaten][field] = 0

        action_nuke_fields = ["enemy_cash_delta", "enemy_mil_delta","self_cash_delta_nuke_cost","self_cash_delta_second_strike",
                              "self_military_delta_second_strike","reward_enemy_no_nuke","reward_enemy_has_nuke", "military_threshold"]
        for field in action_nuke_fields:
            self.data[action_nuke][field] = 0

        # set Initial Conditions
        self.data["ic"] = dict()
        self.data["ic"]["cash"] = 100
        self.data["ic"]["income"] = 0
        self.data["ic"]["military"] = 0
        self.data["ic"]["has_made_threat"] = 0
        self.data["ic"]["has_nukes"] = 0

        self.data["win_reward"] = 1000
        self.data["max_episode_length"] = 1000
        self.data["money_loss_threshold"] = 10
        self.data["invalid_penalty"] = 100

    def get_path_config_folder(self):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.abspath(os.path.join(parent_dir, "../config/v1"))
        return config_dir        

    def save_as_json(self, filename):
        path = os.path.join(self.get_path_config_folder(),filename)
        with open(path, 'w+') as f:
            json.dump(obj=self.data, fp=f, indent=4)

    def load_from_json(self, filename):
        path = os.path.join(self.get_path_config_folder(),filename)
        with open(path, 'r') as f:
            data = json.load(f)

        return data


class MadState_v1:
    '''
    Game State

    Contains all information that an agent needs to know to make a move
    Contains all information that the game env needs to know to process a move

    All information is stored in a numpy array (dtype=int). Can be accessed through
    various property decorators

    A note on semantics: An "obervation" contains information that is available to an agent
                         A "state" contains all information known to the game

    - Player A's
        - cash [int]
        - passive_income [int]
        - military [int]
        - has_made_threat [bool]
        - has_nukes [bool]
    - Player B's
        - cash [int]
        - passive_income [int]
        - military [int]
        - has_made_threat [bool]
        - has_nukes [bool]
    '''

    # Data indexes
    idx_cash_a = 0
    idx_income_a = 1
    idx_military_a = 2
    idx_has_made_threat_a = 3
    idx_has_nukes_a = 4

    idx_cash_b = 5
    idx_income_b = 6
    idx_military_b = 7
    idx_has_made_threat_b = 8
    idx_has_nukes_b = 9

    observation_size = 5

    def __init__(self, config):
        self.data = np.zeros((10), dtype='int')

        # Setup initial conditions
        self.cash_a = config.data["ic"]["cash"]
        self.cash_b = config.data["ic"]["cash"]
        self.income_a = config.data["ic"]["income"]
        self.income_b = config.data["ic"]["income"]
        self.military_a = config.data["ic"]["military"]
        self.military_b = config.data["ic"]["military"]
        self.has_made_threat_a = config.data["ic"]["has_made_threat"]
        self.has_made_threat_b = config.data["ic"]["has_made_threat"]
        self.has_nukes_a = config.data["ic"]["has_nukes"]
        self.has_nukes_b = config.data["ic"]["has_nukes"]

    def swap_agents(self):
        '''
        Swaps the state of the agents
        '''
        tmp = copy.deepcopy(self.data[0:5])
        self.data[0:5] = self.data[5:10]
        self.data[5:10] = tmp

    # Property decoratory
    # Observations for each agent
    @property
    def observation_a(self):
        return self.data[0:5]

    @property
    def observation_b(self):
        return self.data[5:10]

    # Player A
    # Cash A
    @property
    def cash_a(self):
        return self.data[self.idx_cash_a].astype(int)

    @cash_a.setter
    def cash_a(self, x):
        self.data[self.idx_cash_a] = x

    # Income A
    @property
    def income_a(self):
        return self.data[self.idx_income_a].astype(int)

    @income_a.setter
    def income_a(self, x):
        self.data[self.idx_income_a] = x

    # Military A
    @property
    def military_a(self):
        return self.data[self.idx_military_a].astype(int)

    @military_a.setter
    def military_a(self, x):
        self.data[self.idx_military_a] = x

    # Has Made Threat A
    @property
    def has_made_threat_a(self):
        return self.data[self.idx_has_made_threat_a].astype(int)

    @has_made_threat_a.setter
    def has_made_threat_a(self, x):
        self.data[self.idx_has_made_threat_a] = x

    # Has Nukes A
    @property
    def has_nukes_a(self):
        return self.data[self.idx_has_nukes_a].astype(int)

    @has_nukes_a.setter
    def has_nukes_a(self, x):
        self.data[self.idx_has_nukes_a] = x

    # Player B
    # Cash B
    @property
    def cash_b(self):
        return self.data[self.idx_cash_b].astype(int)

    @cash_b.setter
    def cash_b(self, x):
        self.data[self.idx_cash_b] = x

    # Income B
    @property
    def income_b(self):
        return self.data[self.idx_income_b].astype(int)

    @income_b.setter
    def income_b(self, x):
        self.data[self.idx_income_b] = x

    # Military B
    @property
    def military_b(self):
        return self.data[self.idx_military_b].astype(int)

    @military_b.setter
    def military_b(self, x):
        self.data[self.idx_military_b] = x

    # Has Made Threat B
    @property
    def has_made_threat_b(self):
        return self.data[self.idx_has_made_threat_b].astype(int)

    @has_made_threat_b.setter
    def has_made_threat_b(self, x):
        self.data[self.idx_has_made_threat_b] = x

    # Has Nukes B
    @property
    def has_nukes_b(self):
        return self.data[self.idx_has_nukes_b].astype(int)

    @has_nukes_b.setter
    def has_nukes_b(self, x):
        self.data[self.idx_has_nukes_b] = x

    def __repr__(self):
        repr_str = ''
        exclude_list = ['__', 'ic', 'idx', 'data', 'observation', 'swap']
        for attr in dir(self):
            is_excluded = False
            for e in exclude_list:
                if e in attr:
                    is_excluded = True
                    break

            if is_excluded:
                continue

            attr_value = getattr(self, attr)
            repr_str += "MadState_v1.{:20} = {:>5}\n".format(
                attr, attr_value)

        return repr_str


class MadAction_v1:
    '''
    Game Action

    Invest in economy
    Invest in military
    Threaten to attack
    Attack
    Nuke

    Is contructed from a 1-hot numpy array. The index of the 1
    indicates the action to be taken
    '''

    idx_invest_economy = 0
    idx_invest_military = 1
    idx_threaten = 2
    idx_attack = 3
    idx_nuke = 4

    action_size = 5

    action_invest_economy = "Invest Economy"
    action_invest_military = "Invest Military"
    action_threaten = "Threaten"
    action_attack = "Attack"
    action_nuke = "Nuke"
    action_strings = [action_invest_economy,
                      action_invest_military,
                      action_threaten,
                      action_attack,
                      action_nuke]

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be of type numpy.ndarray")

        if not data.shape == (MadAction_v1.action_size,):
            raise ValueError(f"Data must be of shape ({MadAction_v1.action_size},)")

        if not (np.sum(data == 0) == MadAction_v1.action_size-1 and np.sum(data == 1) == 1):
            raise ValueError("Data must be one hot")

        self.data = data

    def action_invest_economy_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        # action_invest_eco_fields = ["income_delta", "cash_delta", "log_coefficient", "reward_offset", "min_reward"]
        reward = 0#max(1, C.data[]["log_coefficient"] * np.log(S))
        done = False
        winner = False
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_invest_economy]
        return reward, done, winner, info

    def action_invest_military_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        done = False
        winner = False
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_invest_military]
        return reward, done, winner, info

    def action_threaten_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        done = False
        winner = False
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_threaten]
        return reward, done, winner, info

    def action_attack_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        done = False
        winner = False
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_attack]
        return reward, done, winner, info

    def action_nuke_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        done = False
        winner = False
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_nuke]
        return reward, done, winner, info

    def get_dynamics_fn(self):
        dynamics = [self.action_invest_economy_dynamics,
                    self.action_invest_military_dynamics,
                    self.action_threaten_dynamics,
                    self.action_attack_dynamics,
                    self.action_nuke_dynamics]
        return dynamics[self.action_idx]

    def apply_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        self.get_dynamics_fn()(S,C)
                    
    @property
    def action_idx(self):
        return np.where(self.data == 1)[0][0]

    @property
    def action_str(self):
        return self.action_strings[self.action_idx]

    def __repr__(self):
        repr_str = ''
        exclude_list = ['__', 'idx', 'data', 'action']
        for attr in dir(self):
            is_excluded = False
            for e in exclude_list:
                if e in attr:
                    is_excluded = True
                    break

            if is_excluded:
                continue

            attr_value = getattr(self, attr)
            repr_str += "MadAction_v1.{:20} = {:>5}\n".format(
                attr, attr_value)

        return repr_str
    
    def max_str_len(self):
        max_len = int(0)
        for string in self.action_strings:
            max_len = max(len(string), max_len)
        return max_len
            


class MadEnv_v1(gym.Env):
    '''
    Game environment

    Unlike other gym environments, step returns a dictionary with an observation for each agent
    '''

    metadata = {'render.modes': ['human']}
    agent_a = 'Agent A'
    agent_b = 'Agent B'

    def __init__(self):
        self.show_bar = False
        self.bar = None
        self.bar_episode = 0
        self.turn_count = 0
        self.config_path = None
        self.reset()

    def set_config_path(self, path):
        self.config_path = path
        self.reset()

    def step(self, A):
        A = MadAction_v1(A)

        # Separate player states
        if self.current_player == self.agent_a:
            self.A_action = A
            reward, done, winner, info = A.apply_dynamics(self.S,self.config)
        else:
            self.B_action = A
            self.S.swap_agents()  # Game dynamics assumes playing agent is A
            # Since B is playing, swap agents so that B is A and A is B
            # then swap back
            reward, done, winner, info = A.apply_dynamics(self.S,self.config)
            self.S.swap_agents()

            # TODO: Winner is always agent a because it is
            # determined in game dynamics. Fix this.
            # temp solution to fix Agent A always winning
            if (winner == self.agent_a):
                winner = self.agent_b
            elif (winner == self.agent_b):
                winner = self.agent_a

        observation = dict()
        observation[self.agent_a] = self.S.observation_a
        observation[self.agent_b] = self.S.observation_b


        self.turn_count += 1
        if self.bar is not None:
            L = A.max_str_len()
            postfix = f"A_ac={self.A_action.action_str:>{L}}, B_ac={self.B_action.action_str:>{L}}, winner={winner}"
            self.bar.set_postfix_str(postfix)
            self.bar.update()
        if self.turn_count >= self.config.data["max_episodes_length"]:
            done = True
            if self.bar is not None:
                self.bar.close()
                self.bar = None

        info['turn_count'] = self.turn_count
        info['action'] = A
        info['winner'] = winner
        info['player'] = self.current_player

        self.change_playing_agent()

        return observation, reward, done, info

    def change_playing_agent(self):
        if self.current_player == self.agent_a:
            self.current_player = self.agent_b
        else:
            self.current_player = self.agent_a

    def set_show_bar(self, show=True, e = 0):
        self.show_bar = show
        self.bar_episode = e

    def reset(self):
        self.turn_count = 0
        self.current_player = self.agent_a
        self.config = MadGameConfig_v1(self.config_path)
        self.S = MadState_v1(self.config)
        observation = dict()
        observation[self.agent_a] = self.S.observation_a
        observation[self.agent_b] = self.S.observation_b

        if self.bar is not None:
            self.bar.close()
            self.bar = None

        if self.show_bar:
            self.bar = tqdm(
                total=self.config.data["max_episodes"], 
                dynamic_ncols=True, 
                leave=True,
                # position=0, 
                desc=f'MAD Episode {self.bar_episode:5d}'
            )

        return observation

    def render(self, mode='human', close=False):
        print(self.S)

    @property
    def observation_size(self):
        return MadState_v1.observation_size

    @property
    def action_size(self):
        return MadAction_v1.action_size
