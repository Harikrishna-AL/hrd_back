from thermal_regulation.envs import SwimmerThermalRegulationEnv, SwimmerSmallThermalRegulationEnv


class TestEnv:

    def test_instance(self):
        env = SwimmerThermalRegulationEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = SwimmerThermalRegulationEnv()
        env.reset()

    def test_run_env(self):
        env = SwimmerThermalRegulationEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = SwimmerThermalRegulationEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 8 + 10 + 2
        assert len(env.action_space.high) == 2 + 1  # motor action + evaporative action
        assert len(obs) == 8 + 10 + 2
        assert len(env.action_space.sample()) == 2 + 1
