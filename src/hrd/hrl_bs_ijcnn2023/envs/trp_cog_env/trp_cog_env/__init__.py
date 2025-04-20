from gym.envs.registration import register

register(
    id='AntTRP-v0',
    entry_point='trp_cog_env.envs:AntTwoResourceEnv',
)

register(
    id='SmallAntTRP-v0',
    entry_point='trp_cog_env.envs:AntSmallTwoResourceEnv',
)

register(
    id='SensorAntTRP-v0',
    entry_point='trp_cog_env.envs:SensorAntTwoResourceEnv',
)

register(
    id='SmallSensorAntTRP-v0',
    entry_point='trp_cog_env.envs:SensorAntSmallTwoResourceEnv',
)

register(
    id='LowGearAntTRP-v0',
    entry_point='trp_cog_env.envs:LowGearAntTwoResourceEnv',
)

register(
    id='SmallLowGearAntTRP-v0',
    entry_point='trp_cog_env.envs:LowGearAntSmallTwoResourceEnv',
)

register(
    id='SnakeTRP-v0',
    entry_point='trp_cog_env.envs:SnakeTwoResourceEnv',
)

register(
    id='SmallSnakeTRP-v0',
    entry_point='trp_cog_env.envs:SnakeSmallTwoResourceEnv',
)

register(
    id='SwimmerTRP-v0',
    entry_point='trp_cog_env.envs:SwimmerTwoResourceEnv',
)

register(
    id='SmallSwimmerTRP-v0',
    entry_point='trp_cog_env.envs:SwimmerSmallTwoResourceEnv',
)

register(
    id='HumanoidTRP-v0',
    entry_point='trp_cog_env.envs:HumanoidTwoResourceEnv',
)

register(
    id='SmallHumanoidTRP-v0',
    entry_point='trp_cog_env.envs:HumanoidSmallTwoResourceEnv',
)

register(
    id='RealAntTRP-v0',
    entry_point='trp_cog_env.envs:RealAntTwoResourceEnv',
)

register(
    id='SmallRealAntTRP-v0',
    entry_point='trp_cog_env.envs:RealAntSmallTwoResourceEnv',
)
