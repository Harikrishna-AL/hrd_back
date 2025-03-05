from thermal_regulation.envs import SensorAntThermalRegulationEnv, SensorAntSmallThermalRegulationEnv


class TestEnv:

    def test_instance(self):
        env = SensorAntThermalRegulationEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = SensorAntThermalRegulationEnv()
        env.reset()

    def test_run_env(self):
        env = SensorAntThermalRegulationEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = SensorAntThermalRegulationEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 10 + 10 + 2  # ant proprioception/range sensor/food sensor/interoception
        assert len(env.action_space.high) == 8 + 1  # motor action + evaporative action
        assert len(obs) == 27 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 8 + 1
