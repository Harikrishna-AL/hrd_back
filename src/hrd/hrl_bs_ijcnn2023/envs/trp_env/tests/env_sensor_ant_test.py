import numpy as np
from pytest import approx

from trp_env.envs.ant_sensor_trp_env import SensorAntTwoResourceEnv, SensorAntSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = SensorAntTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = SensorAntTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = SensorAntTwoResourceEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = SensorAntTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 27 + 10 + 10 + 10 + 2
        assert len(env.action_space.high) == 8
        assert len(obs) == 27 + 10 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 8
