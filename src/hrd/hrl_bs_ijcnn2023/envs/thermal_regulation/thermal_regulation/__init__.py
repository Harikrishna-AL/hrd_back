from gym.envs.registration import register

register(
    id='AntTHR-v3',
    entry_point='thermal_regulation.envs:AntThermalRegulationEnv',
)
register(
    id='SmallAntTHR-v3',
    entry_point='thermal_regulation.envs:AntSmallThermalRegulationEnv',
)

register(
    id='SensorAntTHR-v3',
    entry_point='thermal_regulation.envs:SensorAntThermalRegulationEnv',
)
register(
    id='SmallSensorAntTHR-v3',
    entry_point='thermal_regulation.envs:SensorAntSmallThermalRegulationEnv',
)

register(
    id='LowGearAntTHR-v3',
    entry_point='thermal_regulation.envs:LowGearAntThermalRegulationEnv',
)
register(
    id='SmallLowGearAntTHR-v3',
    entry_point='thermal_regulation.envs:LowGearAntSmallThermalRegulationEnv',
)

register(
    id='SnakeTHR-v3',
    entry_point='thermal_regulation.envs:SnakeThermalRegulationEnv',
)
register(
    id='SmallSnakeTHR-v3',
    entry_point='thermal_regulation.envs:SnakeSmallThermalRegulationEnv',
)

register(
    id='SwimmerTHR-v3',
    entry_point='thermal_regulation.envs:SwimmerThermalRegulationEnv',
)
register(
    id='SmallSwimmerTHR-v3',
    entry_point='thermal_regulation.envs:SwimmerSmallThermalRegulationEnv',
)
