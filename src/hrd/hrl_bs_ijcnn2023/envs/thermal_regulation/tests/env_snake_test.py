from thermal_regulation.envs import SnakeThermalRegulationEnv, SnakeSmallThermalRegulationEnv


class TestEnv:

    def test_instance(self):
        env = SnakeThermalRegulationEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = SnakeThermalRegulationEnv()
        env.reset()

    def test_run_env(self):
        env = SnakeThermalRegulationEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = SnakeThermalRegulationEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 12 + 10 + 2  # 17 + 10 + 2 if non-ego-centric observation
        assert len(env.action_space.high) == 4 + 1  # motor action + evaporative action
        assert len(obs) == 12 + 10 + 2 # 17 + 10 + 2 if non-ego-centric observation
        assert len(env.action_space.sample()) == 4 + 1
