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

    # Property decorators

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
        exclude_list = ['__', 'ic', 'idx', 'data']
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


class MadEnv_v0(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.S = MadState_v0()

    def step(self, A):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        print(self.S)
