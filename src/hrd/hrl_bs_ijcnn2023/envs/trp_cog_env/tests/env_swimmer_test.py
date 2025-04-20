import numpy as np
from pytest import approx

from trp_env.envs.swimmer_trp_env import SwimmerTwoResourceEnv, SwimmerSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = SwimmerTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = SwimmerTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = SwimmerTwoResourceEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = SwimmerTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 8 + 10 + 10 + 2
        assert len(env.action_space.high) == 2
        assert len(obs) == 8 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 2
