from gym.envs.registration import register

register(
    id='mad-v0',
    entry_point='gym_mad.envs:MadEnv_v0',
)
register(
    id='mad-v1',
    entry_point='gym_mad.envs:MadEnv_v1',
)
