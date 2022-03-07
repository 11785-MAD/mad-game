import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import json
import os


class MadGameConfig_v0:
    '''
    All game hyperparameters go in this config class
    '''

    def __init__(self, filename=None):
        '''
        This function automatically finds path to config folder
        '''
        self.data = dict()
        for action in MadAction_v0.action_strings:
            self.data[action] = dict()
            self.data[action]["player"] = dict()
            self.data[action]["enemy"] = dict()
            self.data[action]["penalty"] = dict()
            self.data[action]["player"]["cash_threshold"] = 0
            self.data[action]["player"]["military_threshold"] = 0
            self.data[action]["player"]["requires_threat"] = 0
            self.data[action]["player"]["cash_delta"] = 0
            self.data[action]["player"]["income_delta"] = 0
            self.data[action]["player"]["military_delta"] = 0
            self.data[action]["player"]["has_made_threat_set"] = 0
            self.data[action]["player"]["has_made_threat_clear"] = 0
            self.data[action]["enemy"]["cash_delta"] = 0
            self.data[action]["enemy"]["income_delta"] = 0
            self.data[action]["enemy"]["military_delta"] = 0
            self.data[action]["penalty"]["insufficient_cash"] = 0
            self.data[action]["penalty"]["insufficient_military"] = 0
            self.data[action]["penalty"]["insufficient_threat"] = 0
            self.data[action]["reward"] = 0

        self.data["ic"] = dict()
        self.data["ic"]["cash"] = 10
        self.data["ic"]["income"] = 10
        self.data["ic"]["military"] = 10
        self.data["ic"]["has_made_threat"] = 10
        self.data["ic"]["has_nukes"] = 0

        self.data["win_reward"] = 0

    def get_path_config_folder(self):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.abspath(os.path.join(parent_dir, "../config/v0"))
        return config_dir


    def save_as_json(self, filename):
        path = os.path.join(self.get_path_config_folder(),filename)
        with open(path, 'w+') as f:
            json.dump(obj=self.data, fp=f, indent=4)

    def load_from_json(self, filename):
        path = os.path.join(self.get_path_config_folder(),filename)
        with open(path, 'r') as f:
            data = json.load(f)
            print(data)


class MadState_v0:
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
        tmp = self.data[0:5]
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
            repr_str += "MadState_v0.{:20} = {:>5}\n".format(
                attr, attr_value)

        return repr_str


class MadAction_v0:
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

        if not data.shape == (5,):
            raise ValueError("Data must be of shape (5,)")

        if not (np.sum(data == 0) == 4 and np.sum(data == 1) == 1):
            raise ValueError("Data must be one hot")

        self.data = data

    @property
    def invest_economy(self):
        return self.data[self.idx_invest_economy].astype(bool)

    @property
    def invest_military(self):
        return self.data[self.idx_invest_military].astype(bool)

    @property
    def threaten(self):
        return self.data[self.idx_threaten].astype(bool)

    @property
    def attack(self):
        return self.data[self.idx_attack].astype(bool)

    @property
    def nuke(self):
        return self.data[self.idx_nuke].astype(bool)

    @property
    def action_str(self):

        return self.action_strings[np.where(self.data == 1)[0][0]]

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
            repr_str += "MadAction_v0.{:20} = {:>5}\n".format(
                attr, attr_value)

        return repr_str


class MadEnv_v0(gym.Env):
    '''
    Game environment

    Unlike other gym environments, step returns a dictionary with an observation for each agent
    '''

    metadata = {'render.modes': ['human']}
    agent_a = 'Agent A'
    agent_b = 'Agent B'

    def __init__(self):
        self.config = MadGameConfig_v0()
        self.reset()

    def step(self, A):
        A = MadAction_v0(A)

        # Seperate player states
        if self.current_player == self.agent_a:
            reward, done, winner = self.game_dynamics(A)
        else:
            self.S.swap_agents()  # Game dynamics assumes playing agent is A
            # Since be is playing, swap agents so that b is a and a is b
            # then swap back
            reward, done, winner = self.game_dynamics(A)
            self.S.swap_agents()

        observation = dict()
        observation[self.agent_a] = self.S.observation_a
        observation[self.agent_b] = self.S.observation_b

        self.change_playing_agent()

        info = dict()
        info['action'] = A
        info['winner'] = winner

        return observation, reward, done, info

    def game_dynamics(self, action):
        '''
        Assumes the playing agent is agent_a
        and the waiting agent is agent_b

        Return rewards, done, winner
        Updates state in place
        '''
        # TODO move these numbers into a config file
        action_str = action.action_str
        action_dict = self.config.data[action_str]

        reward = 0
        done = False
        winner = False

        # Check to see if player has resources to execute action
        has_enough_cash = self.S.cash_a >= action_dict["player"][
            "cash_threshold"]
        if not has_enough_cash:
            reward += action_dict["penalty"]["insufficient_cash"]
            return reward, done, winner

        has_enough_military = self.S.cash_a >= action_dict["player"][
            "military_threshold"]
        if not has_enough_military:
            reward += action_dict["penalty"]["insufficient_military"]
            return reward, done, winner

        if action_dict["player"]["requires_threat"] and not self.S.has_made_threat_a:
            reward += action_dict["penalty"]["insufficient_threat"]
            return reward, done, winner

        # Player has enough resources for the action, execute the action
        # Update player resources
        self.S.cash_a = max(
            0,
            self.S.cash_a +
            action_dict["player"]["cash_delta"])
        self.S.income_a = max(
            0,
            self.S.income_a +
            action_dict["player"]["income_delta"])
        self.S.military_a = max(
            0,
            self.S.military_a +
            action_dict["player"]["military_delta"])

        # Set or clear threat
        if action_dict["player"]["has_made_threat_set"]:
            self.S.has_made_threat_a = 1

        if action_dict["player"]["has_made_threat_clear"]:
            self.S.has_made_threat_a = 0

        # Update enemy resources
        self.S.cash_b = max(
            0,
            self.S.cash_b +
            action_dict["enemy"]["cash_delta"])
        self.S.income_b = max(
            0,
            self.S.income_b +
            action_dict["enemy"]["income_delta"])
        self.S.military_b = max(
            0,
            self.S.military_b +
            action_dict["enemy"]["military_delta"])

        # set or clear has nukes
        self.has_nukes_a = self.S.military_a >= self.config.data[
                MadAction_v0.action_strings[MadAction_v0.idx_nuke]]["player"]["military_threshold"]

        self.has_nukes_b = self.S.military_b >= self.config.data[
                MadAction_v0.action_strings[MadAction_v0.idx_nuke]]["player"]["military_threshold"]


        reward = action_dict["reward"]

        # If player has no money, no income, and not enough military to attack
        # then they have lost
        if self.S.cash_a == 0 and self.S.income_a == 0 and self.S.military_a < self.config.data[
                MadAction_v0.action_strings[MadAction_v0.idx_attack]]["player"]["cash_threshold"]:
            done = True
            winner = self.agent_b
            reward += self.config.data["win_reward"]
            return reward, done, winner

        if self.S.cash_b == 0 and self.S.income_b == 0 and self.S.military_b < self.config.data[
                MadAction_v0.action_strings[MadAction_v0.idx_attack]]["player"]["cash_threshold"]:
            done = True
            winner = self.agent_a
            reward += self.config.data["win_reward"]
            return reward, done, winner

        return reward, done, winner

    def change_playing_agent(self):
        if self.current_player == self.agent_a:
            self.current_player = self.agent_b
        else:
            self.current_player = self.agent_a

    def reset(self):
        self.current_player = self.agent_a
        self.S = MadState_v0(self.config)
        observation = dict()
        observation[self.agent_a] = self.S.observation_a
        observation[self.agent_b] = self.S.observation_b

        return observation

    def render(self, mode='human', close=False):
        print(self.S)

    @property
    def observation_size(self):
        return MadState_v0.observation_size

    @property
    def action_size(self):
        return MadAction_v0.action_size
