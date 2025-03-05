import pytest
import numpy as np

from pytest import approx

from trp_env.envs.two_resource_env import FoodClass
from trp_env.envs.realant_trp_env import RealAntTwoResourceEnv, RealAntSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = RealAntTwoResourceEnv()
        
        env = RealAntSmallTwoResourceEnv()

    def test_reset_env(self):
        env = RealAntSmallTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = RealAntSmallTwoResourceEnv(show_sensor_range=True)#, sensor_range=0.3, n_bins=8)
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())

    def test_dim(self):
        env = RealAntTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 29 + 10 + 10 + 2
        assert len(env.action_space.high) == 8
        assert len(obs) == 29 + 10 + 10 + 2
        assert len(env.action_space.sample()) == 8

    def test_dim_small(self):
        env = RealAntSmallTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 29 + 20 + 20 + 2
        assert len(env.action_space.high) == 8
        assert len(obs) == 29 + 20 + 20 + 2
        assert len(env.action_space.sample()) == 8
