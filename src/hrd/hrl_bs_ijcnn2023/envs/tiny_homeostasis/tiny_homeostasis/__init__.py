from gym.envs.registration import register

register(
    id='LineHomeostatic-v0',
    entry_point='tiny_homeostasis.envs:LineEnv',
)

register(
    id='FieldHomeostatic-v0',
    entry_point='tiny_homeostasis.envs:FieldEnv',
)
