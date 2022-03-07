import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class MadState_v0:
    '''
    Game State

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

    # Inital conditions
    ic_cash = 10
    ic_income = 10
    ic_military = 10
    ic_has_made_threat = 0
    ic_has_nukes = 0

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

    def __init__(self):
        self.data = np.zeros((10), dtype='int')

        # Setup initial conditions
        self.cash_a = self.ic_cash
        self.cash_b = self.ic_cash
        self.income_a = self.ic_income
        self.income_b = self.ic_income
        self.military_a = self.ic_military
        self.military_b = self.ic_military
        self.has_made_threat_a = self.ic_has_made_threat
        self.has_made_threat_b = self.ic_has_made_threat

    # Property decoratory
    # Observations for each agent
    @property
    def observation_a(self):
        return self.data[0:5]

    @property
    def observation_b(self):
        return self.data[5:10]

    # State of each agent
    @property
    def state_a(self):
        return self.data[0:5]

    @state_a.setter
    def state_a(self, x):
        self.data[0:5] = x

    @property
    def state_b(self):
        return self.data[5:10]

    @state_b.setter
    def state_b(self, x):
        self.data[5:10] = x

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
        exclude_list = ['__', 'ic', 'idx', 'data', 'observation', 'state']
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
    '''

    idx_invest_economy = 0
    idx_invest_military = 1
    idx_threaten = 2
    idx_attack = 3
    idx_nuke = 4

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

    def __repr__(self):
        repr_str = ''
        exclude_list = ['__', 'idx', 'data']
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
    metadata = {'render.modes': ['human']}
    agent_a = 'Agent A'
    agent_b = 'Agent B'

    def __init__(self):
        self.reset()

    def step(self, A):
        A = MadAction_v0(A)
        print(A)

        # Seperate player states
        if self.current_player == self.agent_a:
            playing_agent_state = self.S.state_a
            waiting_agent_state = self.S.state_b
        else:
            playing_agent_state = self.S.state_b
            waiting_agent_state = self.S.state_a

        # Execute game dynamics
        new_playing_agent_state, new_waiting_agent_state = self.game_dynamics(
                                                                              playing_agent_state,
                                                                              waiting_agent_state,
                                                                              A)

        # Update Player States
        if self.current_player == self.agent_a:
            self.S.state_a = new_playing_agent_state
            self.S.state_b = new_waiting_agent_state
        else:
            self.S.state_a = new_waiting_agent_state
            self.S.state_b = new_playing_agent_state

        observation = dict()
        observation[self.agent_a] = self.S.observation_a
        observation[self.agent_b] = self.S.observation_b
        reward, done = self.get_reward()
        info = dict()

        self.change_playing_agent()
        info['current_player'] = self.current_player

        return observation, reward, done, info

    def game_dynamics(self, playing_agent_state, waiting_agent_state, action):
        return playing_agent_state, waiting_agent_state

    def get_reward(self):
        # TODO
        reward = 0.0
        done = False
        return reward, done

    def change_playing_agent(self):
        if self.current_player == self.agent_a:
            self.current_player = self.agent_b
        else:
            self.current_player = self.agent_a

    def reset(self):
        self.current_player = self.agent_a
        self.S = MadState_v0()

    def render(self, mode='human', close=False):
        print(self.S)
