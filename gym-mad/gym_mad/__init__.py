from gym.envs.registration import register

register(
    id='mad-v0',
    entry_point='gym_mad.envs:MadEnv-v0',
)
