from thermal_regulation.envs import AntThermalRegulationEnv, AntSmallThermalRegulationEnv


class TestEnv:

    def test_instance(self):
        env = AntThermalRegulationEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = AntThermalRegulationEnv()
        env.reset()

    def test_run_env(self):
        env = AntThermalRegulationEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = AntThermalRegulationEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 10 + 2
        assert len(env.action_space.high) == 8 + 1  # motor action + evaporative action
        assert len(obs) == 27 + 10 + 2
        assert len(env.action_space.sample()) == 8 + 1

    def test_dim_small(self):
        env = AntSmallThermalRegulationEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 20 + 2
        assert len(env.action_space.high) == 8 + 1
        assert len(obs) == 27 + 20 + 2
        assert len(env.action_space.sample()) == 8 + 1