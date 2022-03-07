import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class MadState_v0:
    '''
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
        self.data = np.zeros((10),dtype='int')

    @property
    def cash_a(self):
        return self.data[self.idx_cash_a].astype(int)

    @cash_a.setter
    def cash_a(self, x):
        self.data[self.idx_cash_a] = x

    def __repr__(self):
        repr_str = ''
        exclude_list = ['__','ic','idx','data']
        for attr in dir(self):
            is_excluded = False
            for e in exclude_list:
                if e in attr:
                    is_excluded = True
                    break

            if is_excluded:
                continue

            print(attr)
            print(type(attr))
            attr_value = getattr(self, attr)
            print(attr_value)
            print(type(attr_value))
            print(repr_str)
            print(type(repr_str))
            repr_str += "MadState_v0.{:20} = {:>5}\n".format(
                attr, attr_value)

        return repr_str


class MadEnv_v0(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.state = MadState_v0()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        print(self.state)
