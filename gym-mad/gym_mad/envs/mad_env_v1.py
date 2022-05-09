import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
import json
import os
from tqdm import tqdm
import math


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

        action_invest_eco = MadAction_v1.action_invest_economy
        action_invest_mil = MadAction_v1.action_invest_military
        action_attack = MadAction_v1.action_attack
        action_threaten = MadAction_v1.action_threaten
        action_nuke = MadAction_v1.action_nuke

        # set action-specific attributes
        action_invest_eco_fields = ["income_delta", "cash_delta", "log_coefficient", "log_base", "reward_offset", "min_reward"]
        for field in action_invest_eco_fields:
            self.data[action_invest_eco][field] = 0

        action_invest_mil_fields = ["cash_delta", "military_delta", "log_coefficient","log_epsilon", "military_cash_scale_factor", "military_size_limit"]
        for field in action_invest_mil_fields:
            self.data[action_invest_mil][field] = 0

        action_atk_fields = ["military_threshold", "L_cash", "L_military", "log_coefficient", "log_epsilon"]
        for field in action_atk_fields:
            self.data[action_attack][field] = 0

        action_threaten_fields = ["military_threshold", "reward"]
        for field in action_threaten_fields:
            self.data[action_threaten][field] = 0

        action_nuke_fields = ["military_threshold", "enemy_cash_delta", "enemy_mil_delta","self_cash_delta_nuke_cost","self_cash_delta_second_strike",
                              "self_military_delta_second_strike","reward_enemy_no_nuke","reward_enemy_has_nuke"]
        for field in action_nuke_fields:
            self.data[action_nuke][field] = 0

        # set Initial Conditions
        self.data["ic"] = dict()
        self.data["ic"]["cash"] = 100
        self.data["ic"]["income"] = 0
        self.data["ic"]["military"] = 0
        self.data["ic"]["has_made_threat"] = 0
        self.data["ic"]["has_nukes"] = 0

        # set general params
        self.data["win_reward"] = 1000
        self.data["max_episode_length"] = 1000
        self.data["money_loss_threshold"] = 10
        self.data["invalid_penalty"] = -100
        self.data["over_max_penalty"] = -300
        self.data["max_cash"] = 100000
        self.data["max_income"] = 1000
        self.data["max_military"] = 3000
        self.data["reward_scale"] = 0.001

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

    observation_size = 10

    def __init__(self, config):
        self.config = config
        self.data = np.zeros((10), dtype='float')

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
        S = MadState_v1(self.config)
        S.data = self.data.copy()
        S.cash_a /= S.config.data["max_cash"]
        S.cash_b /= S.config.data["max_cash"]
        S.income_a /= S.config.data["max_income"]
        S.income_b /= S.config.data["max_income"]
        S.military_a /= S.config.data["max_military"]
        S.military_b /= S.config.data["max_military"]
        
        return S.data - 0.5

    @property
    def observation_b(self):
        S = MadState_v1(self.config)
        S.data = self.data.copy()
        S.cash_a /= S.config.data["max_cash"]
        S.cash_b /= S.config.data["max_cash"]
        S.income_a /= S.config.data["max_income"]
        S.income_b /= S.config.data["max_income"]
        S.military_a /= S.config.data["max_military"]
        S.military_b /= S.config.data["max_military"]
        S.swap_agents()
        return S.data - 0.5

    # Player A
    # Cash A
    @property
    def cash_a(self):
        return self.data[self.idx_cash_a].astype(float)

    @cash_a.setter
    def cash_a(self, x):
        self.data[self.idx_cash_a] = x

    # Income A
    @property
    def income_a(self):
        return self.data[self.idx_income_a].astype(float)

    @income_a.setter
    def income_a(self, x):
        self.data[self.idx_income_a] = x

    # Military A
    @property
    def military_a(self):
        return self.data[self.idx_military_a].astype(float)

    @military_a.setter
    def military_a(self, x):
        self.data[self.idx_military_a] = x

    # Has Made Threat A
    @property
    def has_made_threat_a(self):
        return self.data[self.idx_has_made_threat_a].astype(bool)

    @has_made_threat_a.setter
    def has_made_threat_a(self, x):
        self.data[self.idx_has_made_threat_a] = x

    # Has Nukes A
    @property
    def has_nukes_a(self):
        return self.data[self.idx_has_nukes_a].astype(bool)

    @has_nukes_a.setter
    def has_nukes_a(self, x):
        self.data[self.idx_has_nukes_a] = x

    # Player B
    # Cash B
    @property
    def cash_b(self):
        return self.data[self.idx_cash_b].astype(float)

    @cash_b.setter
    def cash_b(self, x):
        self.data[self.idx_cash_b] = x

    # Income B
    @property
    def income_b(self):
        return self.data[self.idx_income_b].astype(float)

    @income_b.setter
    def income_b(self, x):
        self.data[self.idx_income_b] = x

    # Military B
    @property
    def military_b(self):
        return self.data[self.idx_military_b].astype(float)

    @military_b.setter
    def military_b(self, x):
        self.data[self.idx_military_b] = x

    # Has Made Threat B
    @property
    def has_made_threat_b(self):
        return self.data[self.idx_has_made_threat_b].astype(bool)

    @has_made_threat_b.setter
    def has_made_threat_b(self, x):
        self.data[self.idx_has_made_threat_b] = x

    # Has Nukes B
    @property
    def has_nukes_b(self):
        return self.data[self.idx_has_nukes_b].astype(bool)

    @has_nukes_b.setter
    def has_nukes_b(self, x):
        self.data[self.idx_has_nukes_b] = x

    @property
    def str_short(self):
        rv = ''
        rv += '{:.0f} '.format(self.cash_a)
        rv += '{:.0f} '.format(self.income_a)
        rv += '{:.0f} '.format(self.military_a)
        rv += '{:d} '.format(self.has_made_threat_a)
        rv += '{:d} '.format(self.has_nukes_a)
        rv += '{:.0f} '.format(self.cash_b)
        rv += '{:.0f} '.format(self.income_b)
        rv += '{:.0f} '.format(self.military_b)
        rv += '{:d} '.format(self.has_made_threat_b)
        rv += '{:d}'.format(self.has_nukes_b)
        return rv

    def __repr__(self):
        repr_str = ''
        exclude_list = ['__', 'ic', 'idx', 'data', 'observation', 'swap', 'str_short', 'config']
        for attr in dir(self):
            is_excluded = False
            for e in exclude_list:
                if e in attr:
                    is_excluded = True
                    break

            if is_excluded:
                continue

            attr_value = getattr(self, attr)
            repr_str += "MadState_v1.{:20} = {:>5.1f}\n".format(
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

    action_invest_economy   = "Invest Economy"
    action_invest_military  = "Invest Military"
    action_threaten         = "Threaten"
    action_attack           = "Attack"
    action_nuke             = "Nuke"
    action_strings = [action_invest_economy,
                      action_invest_military,
                      action_threaten,
                      action_attack,
                      action_nuke]

    action_invest_economy_short     = "Eco"
    action_invest_military_short    = "Mil"
    action_threaten_short           = "Thr"
    action_attack_short             = "Atk"
    action_nuke_short               = "Nuk"
    action_strings_short = [action_invest_economy_short,
                      action_invest_military_short,
                      action_threaten_short,
                      action_attack_short,
                      action_nuke_short]


    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be of type numpy.ndarray")

        if not data.shape == (MadAction_v1.action_size,):
            raise ValueError(f"Data must be of shape ({MadAction_v1.action_size},)")

        if not (np.sum(data == 0) == MadAction_v1.action_size-1 and np.sum(data == 1) == 1):
            raise ValueError("Data must be one hot")

        self.data = data

    def action_invest_economy_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_invest_economy]

        if S.cash_a < action_dict["cash_threshold"]:
            info["turn_desc"] = 'Player tried to invest in economy but had insufficient cash.'
            reward = C.data["invalid_penalty"]
            return reward, info
        if (S.cash_a >= C.data["max_cash"]):
            info["turn_desc"] = 'Player tried to invest in economy but had over max cash.'
            reward = C.data["over_max_penalty"]
            return reward, info
        if (S.income_a >= C.data["max_income"]):
            info["turn_desc"] = 'Player tried to invest in economy but had over max income.'
            reward = C.data["over_max_penalty"]
            return reward, info


        reward = max(1, action_dict["log_coefficient"] * math.log(S.cash_a, action_dict["log_base"]) + action_dict["reward_offset"])
        S.cash_a += action_dict["cash_delta"]
        S.income_a += action_dict["income_delta"]
        info["turn_desc"] = 'Player invested in economy.'
        return reward, info

    def action_invest_military_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_invest_military]

        if S.cash_a < action_dict["cash_threshold"]:
            info["turn_desc"] = 'Player tried to invest in military but failed.'
            reward = C.data["invalid_penalty"]
            return reward, info
        elif (S.military_a > C.data["max_military"]):
            info["turn_desc"] = 'Player tried to invest in military but had over max military.'
            reward = C.data["over_max_penalty"]
            return reward, info

        ratio = S.cash_a / (S.military_a * action_dict["military_cash_scale_factor"] + action_dict["log_epsilon"])
        if (ratio > 1):
            reward = action_dict["log_coefficient"] * np.log2(ratio)
        else:
            reward = 0
        if S.military_a >= action_dict['military_size_limit']:
            S.military_a *= (action_dict['military_size_limit']/S.military_a)**2
        S.cash_a += action_dict["cash_delta"]
        S.military_a += action_dict["military_delta"]
        info["turn_desc"] = 'Player invested in military.'
        return reward, info

    def action_attack_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_attack]

        if S.cash_a < action_dict["cash_threshold"] or S.military_a < action_dict["military_threshold"]:
            info["turn_desc"] = 'Player tried to attack but failed.'
            reward = C.data["invalid_penalty"]
            return reward, info

        reward = action_dict["log_coefficient"] * (np.log2((S.military_a + action_dict["log_epsilon"]) / (S.military_b + action_dict["log_epsilon"])) + action_dict["log_offset"])

        r_a = S.military_b / (S.military_a + S.military_b)
        r_b = S.military_a / (S.military_a + S.military_b)
        # note: assumes L_cash, L_military are POSITIVE
        S.cash_a -= action_dict["L_cash"] * r_a
        S.military_a -= action_dict["L_military"] * r_a
        S.income_a -= action_dict["L_income"] * r_a
        S.cash_b -= action_dict["L_cash"] * r_b
        S.military_b -= action_dict["L_military"] * r_b
        S.income_b -= action_dict["L_income"] * r_b
        
        if S.military_a < 0:
            unpaid_loss = -S.military_a
            S.cash_a -= unpaid_loss * action_dict["military_cash_scale_factor"]

        if S.military_b < 0:
            unpaid_loss = -S.military_b
            S.cash_b -= unpaid_loss * action_dict["military_cash_scale_factor"]
        
        info["turn_desc"] = 'Player attacked.'
        return reward, info

    def action_threaten_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_threaten]

        if S.cash_a < action_dict["cash_threshold"] or S.military_a < action_dict["military_threshold"] or S.has_made_threat_a:
            info["turn_desc"] = 'Player tried to threaten but failed.'
            reward = C.data["invalid_penalty"]
            return reward, info

        reward = action_dict["reward"]
        S.has_made_threat_a = True
        info["turn_desc"] = 'Player threatened.'
        return reward, info

    def action_nuke_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        reward = 0
        info = dict()
        info["turn_desc"] = ''
        action_dict = C.data[MadAction_v1.action_nuke]

        if S.cash_a < action_dict["cash_threshold"] or S.military_a < action_dict["military_threshold"] or not S.has_made_threat_a:
            info["turn_desc"] = 'Player tried to nuke but failed.'
            reward = C.data["invalid_penalty"]
            return reward, info

        if S.has_made_threat_b:
            reward = action_dict["reward_enemy_has_nuke"]
        else:
            reward = action_dict["reward_enemy_no_nuke"]

        S.cash_a += action_dict["self_cash_delta_nuke_cost"]
        S.cash_b += action_dict["cash_delta"]
        S.military_b += action_dict["mil_delta"]
        S.income_b += action_dict["passive_income_delta"]
        info["turn_desc"] = 'Player dropped a nuke on em.'
        
        if (S.has_nukes_b):
            S.cash_a += action_dict["cash_delta"]
            S.military_a += action_dict["mil_delta"]
            S.income_a += action_dict["passive_income_delta"]
            S.cash_b += action_dict["self_cash_delta_nuke_cost"]
            info["turn_desc"] = 'Player dropped a nuke on em but then enemy dropped a nuke on player.'

        return reward, info

    def get_dynamics_fn(self):
        dynamics = [self.action_invest_economy_dynamics,
                    self.action_invest_military_dynamics,
                    self.action_threaten_dynamics,
                    self.action_attack_dynamics,
                    self.action_nuke_dynamics]
        return dynamics[self.action_idx]

    def apply_dynamics(self, S:MadState_v1, C:MadGameConfig_v1):
        done = False
        winner = None
        reward, info = self.get_dynamics_fn()(S,C)

        # clip values
        
        S.income_a = np.clip(S.income_a, 0, C.data["max_income"])
        S.income_b = np.clip(S.income_b, 0, C.data["max_income"])
        S.military_a = np.clip(S.military_a, 0, C.data["max_military"])
        S.military_b = np.clip(S.military_b, 0, C.data["max_military"])
            
        # increment passive income
        S.cash_a += S.income_a
        S.cash_b += S.income_b
        
        S.cash_a = np.clip(S.cash_a, 0, C.data["max_cash"])
        S.cash_b = np.clip(S.cash_b, 0, C.data["max_cash"])

        # check for nuke
        if S.cash_a >= C.data[MadAction_v1.action_nuke]['cash_threshold'] and S.military_a >= C.data[MadAction_v1.action_nuke]['military_threshold']:
            S.has_nukes_a = True
        else: 
            S.has_nukes_a = False
        if S.cash_b >= C.data[MadAction_v1.action_nuke]['cash_threshold'] and S.military_b >= C.data[MadAction_v1.action_nuke]['military_threshold']:
            S.has_nukes_b = True
        else: 
            S.has_nukes_b = False
            
        # check for Winner
        a_lost = S.cash_a <= 0 and S.military_a <= 0 and S.income_a <= 0
        b_lost = S.cash_b <= 0 and S.military_b <= 0 and S.income_b <= 0
        if a_lost and b_lost:
            done = True
        elif a_lost:
            done = True
            winner = MadEnv_v1.agent_b
        elif b_lost:
            done = True
            winner = MadEnv_v1.agent_a
            reward += C.data["win_reward"]

        reward = reward * C.data["reward_scale"]
        return reward, done, winner, info
                    
    @property
    def action_idx(self):
        return np.where(self.data == 1)[0][0]

    @property
    def action_str(self):
        return self.action_strings[self.action_idx]

    @property
    def action_str_short(self):
        return self.action_strings_short[self.action_idx]
    

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
            
    def max_str_short_len(self):
        max_len = int(0)
        for string in self.action_strings_short:
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
        self.bar_postifx = ''
        self.turn_count = 0
        self.config_path = None
        self.A_action = None
        self.B_action = None
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

            if (winner == self.agent_a):
                winner = self.agent_b
            elif (winner == self.agent_b):
                winner = self.agent_a

        observation = dict()
        observation[self.agent_a] = self.S.observation_a
        observation[self.agent_b] = self.S.observation_b


        self.turn_count += 1
        if self.bar is not None:
            L = A.max_str_short_len()
            if self.A_action is not None and self.B_action is not None:
                postfix = f"{self.bar_postifx}P={self.current_player[-1]}, S=[{self.S.str_short}], ac=[{self.A_action.action_str_short:>{L}}, {self.B_action.action_str_short:>{L}}], W={winner}"
                self.bar.set_postfix_str(postfix)
            self.bar.update()
        if self.turn_count >= self.config.data["max_episode_length"]:
            done = True
            if self.bar is not None:
                self.bar.close()
                self.bar = None

        info['turn_count'] = self.turn_count
        info['action'] = A
        info['winner'] = winner
        info['player'] = self.current_player
        
        if not done: self.change_playing_agent()

        return observation, reward, done, info

    def change_playing_agent(self):
        if self.current_player == self.agent_a:
            self.current_player = self.agent_b
        else:
            self.current_player = self.agent_a

    def set_show_bar(self, show=True, e = 0):
        self.show_bar = show
        self.bar_episode = e

    def set_bar_postfix(self, postfix):
        self.bar_postifx = postfix

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
                total=self.config.data["max_episode_length"], 
                dynamic_ncols=True, 
                leave=True,
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