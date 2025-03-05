import pytest
import numpy as np
from pytest import approx

from tiny_homeostasis.envs import LineEnv


def test_make_env():
	env = LineEnv()
	assert isinstance(env, LineEnv)


# def test_render():
# 	env = LineEnv()
#
# 	env.reset()
#
# 	for _ in range(10):
# 		env.step(env.action_space.sample())
# 		env.render()


def test_action_space():
	env = LineEnv()

	assert np.all(env.action_space.shape == (1,))


def test_obs_space():
	env = LineEnv(dim_resource=3)

	obs = env.reset()

	assert np.all(obs.shape == env.observation_space.shape)


def test_intero_dimension():
	env = LineEnv(dim_resource=3)

	env.reset()

	obs, rew, done, info = env.step(env.action_space.sample())

	assert info["interoception"].shape == (3,)


@pytest.mark.parametrize('dim,ans', [
	(2, 1 * 2 + 2 * 2),
	(5, 1 * 2 + 5 * 2),
	(10, 1 * 2 + 10 * 2),
])
def test_obs_dim(dim, ans):
	env = LineEnv(dim_resource=dim)

	obs = env.reset()

	assert obs.shape[0] == ans


def test_default_setting():
	env = LineEnv()

	assert env.dim_resource == 2
	assert env.internal_reset == "random"
	assert np.all(env.internal_random_range == (-1. / 6, 1. / 6))
	assert env._max_episode_steps is np.inf


def test_setting_setpoint():
	env = LineEnv(internal_reset="setpoint")

	env.reset()

	assert np.all(env.resource == np.zeros(2))


def test_get_inlet():
	pass


@pytest.mark.parametrize("setting,expected_mean, expected_var",
						 [
							 ([-1, 1], np.array([0.0, 0.0]), np.array([0.33, 0.33])),
							 ([-0.5, 0.5], np.array([0.0, 0.0]), np.array([1. / 12, 1. / 12])),
							 ([0, 1], np.array([0.5, 0.5]), np.array([1. / 12, 1. / 12])),
						 ])
def test_reset_internal_random_limit(setting, expected_mean, expected_var):
	env = LineEnv(dim_resource=2, internal_reset="random", internal_random_range=setting)

	obs_intero_list = []

	for i in range(3000):
		obs = env.reset()

		obs_intero = obs[-2:]  # interoception

		obs_intero_list.append(obs_intero)

	obs_intero_mean = np.array(obs_intero_list).mean(axis=0)
	obs_intero_var = np.array(obs_intero_list).var(axis=0)

	# Test mean
	np.testing.assert_allclose(actual=obs_intero_mean, desired=expected_mean, atol=0.03)

	# Test var
	np.testing.assert_allclose(actual=obs_intero_var, desired=expected_var, atol=0.02)


@pytest.mark.parametrize('dim', [
	2,
	5,
	10,
])
def test_reward_definition(dim):
	env = LineEnv(dim_resource=dim)

	env.resource_prev = np.ones(dim)
	env.resource = np.zeros(dim)

	r, r_modular, drives = env.get_reward()

	assert r == approx(env.reward_scale * dim, abs=0.0001)
	assert np.all(r_modular == approx(env.reward_scale * 1.0, abs=0.0001))
	assert np.all(drives == approx(0.0, abs=0.0001))


@pytest.mark.parametrize('dim', [
	2,
	5,
	10,
])
def test_intero_obs_position(dim):
	env = LineEnv(dim_resource=dim)

	for _ in range(10):
		env.reset()

		obs, _, _, info = env.step(env.action_space.sample())

		assert np.all(obs[-dim:] == info["interoception"])


@pytest.mark.parametrize('dim', [
	2,
	5,
	10,
])
def test_intero_dim(dim):
	env = LineEnv(dim_resource=dim)
	assert env.dim_intero == dim
