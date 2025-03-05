import pytest
import numpy as np

from pytest import approx

from thermal_regulation.envs import AntThermalRegulationEnv, AntSmallThermalRegulationEnv
from thermal_regulation.envs.thermal_module import Climate
from thermal_regulation.envs.thermal_regulation_env import InteroClass, ObjectClass


def variance_of_uniform(a, b):
    assert a < b
    return (b - a) ** 2 / 12.


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

    def test_instance_not_ego_obs(self):
        env = AntThermalRegulationEnv(
            ego_obs=False,
            no_contact=False,
            sparse=False
        )
        env.reset()

    def test_instance_no_contact(self):
        env = AntThermalRegulationEnv(
            ego_obs=True,
            no_contact=True,
            sparse=False
        )
        env.reset()

    def test_reset_internal_state(self):
        env = AntThermalRegulationEnv(internal_reset="setpoint")
        env.reset()
        env.internal_state = {
            InteroClass.ENERGY: 1.0,
            InteroClass.TEMPERATURE: 1.0,
        }
        for key in InteroClass:
            assert env.internal_state[key] == approx(1.0)

        env.reset()
        initial_internal_state = {
            InteroClass.ENERGY: 0.0,
            InteroClass.TEMPERATURE: 0.0,
        }

        for key in InteroClass:
            assert env.internal_state[key] == initial_internal_state[key]

    def test_reset_if_resource_end(self):
        env = AntThermalRegulationEnv(internal_reset="setpoint")
        env.seed(0)
        env.default_metabolic_update = 0.1
        env.reset()
        while True:
            ob, reward, done, info = env.step(0 * env.action_space.sample())
            if done:
                break
            else:
                intero = ob[-2:]
                assert intero[0] > -env.survival_area and intero[1] > -env.survival_area

        intero = ob[-2:]
        assert intero[0] < -env.survival_area or intero[1] < -env.survival_area
        ob = env.reset()
        intero = ob[-2:]
        assert intero[0] == approx(0) and intero[1] == approx(0)

    def test_run_env(self):
        env = AntThermalRegulationEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())
        env.close()

    def test_render_env(self):
        env = AntSmallThermalRegulationEnv(show_sensor_range=True, n_bins=20, sensor_range=16.)
        for n in range(5):
            env.reset()
            for i in range(10):
                env.step(env.action_space.sample())
                env.render()
        env.close()

    def test_reset(self):
        env = AntThermalRegulationEnv()
        env.seed(0)
        env.reset()
        initial_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        for i in range(1000):
            env.step(env.action_space.sample())

        env.reset()
        reset_robot_pos = env.wrapped_env.get_body_com("torso")[:2].copy()
        np.testing.assert_allclose(actual=reset_robot_pos,
                                   desired=initial_robot_pos,
                                   atol=0.3)

    @pytest.mark.parametrize("setting,expected_mean, expected_var",
                             [
                                 ("setpoint", np.array([0.0, 0.0]), np.array([0.0, 0.0])),
                                 ("random",
                                  np.array([0.0, 0.0]),
                                  np.array([variance_of_uniform(-1 / 6, 1 / 6),
                                            variance_of_uniform(-1 / 6, 1 / 6)])),
                                 ("error_case", None, None)
                             ])
    def test_reset_internal(self, setting, expected_mean, expected_var):
        if setting != "error_case":
            env = AntThermalRegulationEnv(internal_reset=setting)
        else:
            with pytest.raises(ValueError):
                AntThermalRegulationEnv(internal_reset=setting)
            return
        env.seed(0)

        obs_intero_list = []

        for i in range(1000):
            obs = env.reset()

            obs_intero = obs[-2:]  # interoception

            obs_intero_list.append(obs_intero)

        obs_intero_mean = np.array(obs_intero_list).mean(axis=0)
        obs_intero_var = np.array(obs_intero_list).var(axis=0)

        # Test mean
        np.testing.assert_allclose(actual=obs_intero_mean,
                                   desired=expected_mean,
                                   atol=0.06)

        # Test var
        np.testing.assert_allclose(actual=obs_intero_var,
                                   desired=expected_var,
                                   atol=0.06)

    @pytest.mark.parametrize("setting_e,setting_t,expected_mean, expected_var",
                             [
                                 ([-1, 1], [34, 42], np.array([0.0, 0.0]), np.array([0.33, 0.33])),
                                 ([-0.5, 0.5], [36, 40], np.array([0.0, 0.0]), np.array([1. / 12, 1. / 12])),
                                 ([0, 1], [38, 42], np.array([0.5, 0.5]), np.array([1. / 12, 1. / 12])),
                             ])
    def test_reset_internal_random_limit(self, setting_e, setting_t, expected_mean, expected_var):
        env = AntThermalRegulationEnv(internal_reset="random",
                                      energy_random_range=setting_e,
                                      temperature_random_range=setting_t)
        env.seed(0)

        obs_intero_list = []

        for i in range(2000):
            obs = env.reset()

            obs_intero = obs[-2:]  # interoception

            obs_intero_list.append(obs_intero)

        obs_intero_mean = np.array(obs_intero_list).mean(axis=0)
        obs_intero_var = np.array(obs_intero_list).var(axis=0)

        # Test mean
        np.testing.assert_allclose(actual=obs_intero_mean,
                                   desired=expected_mean,
                                   atol=0.03)

        # Test var
        np.testing.assert_allclose(actual=obs_intero_var,
                                   desired=expected_var,
                                   atol=0.02)

    @pytest.mark.parametrize("reward_setting,expected,param",
                             [
                                 ("homeostatic", -2, None),
                                 ("homeostatic", -2, 0.5),  # Reward bias should be ignored
                                 ("homeostatic_shaped", +2, None),
                                 ("one", 0, None),
                                 ("homeostatic_biased", -1.5, 0.5),
                                 ("something_else", None, None),
                             ])
    def test_reward_definition(self, reward_setting, expected, param):
        env = AntThermalRegulationEnv(reward_setting=reward_setting, coef_main_rew=1.0, reward_bias=param)
        env.seed(0)

        action = env.action_space.sample() * 0

        env.prev_interoception = np.array([1, 1])
        env.internal_state = {
            InteroClass.ENERGY: 0.0,
            InteroClass.TEMPERATURE: 0.0,
        }

        if reward_setting != "something_else":
            rew, info = env.get_reward(reward_setting, action, False)
            assert rew == approx(expected, abs=0.0001)
        else:
            with pytest.raises(ValueError):
                env.get_reward(reward_setting, action, False)

        if reward_setting == "one":
            env.internal_state = {
                InteroClass.ENERGY: -0.99999,
                InteroClass.TEMPERATURE: -0.999999,
            }
            _, reward, done, _ = env.step(action)

            assert done
            assert reward == approx(-1.0, abs=0.0001)

    @pytest.mark.parametrize("reward_setting,expected,param",
                             [
                                 ("homeostatic", -2, None),
                                 ("homeostatic", -2, 0.5),  # Reward bias should be ignored
                                 ("homeostatic_shaped", +2, None),
                                 ("one", 0, None),
                                 ("homeostatic_biased", -1.5, 0.5),
                                 ("something_else", None, None),
                             ])
    def test_reward_definition_small(self, reward_setting, expected, param):
        env = AntSmallThermalRegulationEnv(reward_setting=reward_setting, coef_main_rew=1.0, reward_bias=param)
        env.seed(0)

        action = env.action_space.sample() * 0

        env.prev_interoception = np.array([1, 1])
        env.internal_state = {
            InteroClass.ENERGY: 0.0,
            InteroClass.TEMPERATURE: 0.0,
        }

        if reward_setting != "something_else":
            rew, info = env.get_reward(reward_setting, action, False)
            assert rew == approx(expected, abs=0.0001)
        else:
            with pytest.raises(ValueError):
                env.get_reward(reward_setting, action, False)

        if reward_setting == "one":
            env.internal_state = {
                InteroClass.ENERGY: -0.999999,
                InteroClass.TEMPERATURE: -0.999999,
            }
            _, reward, done, _ = env.step(action)

            assert done
            assert reward == approx(-1.0, abs=0.0001)

    def test_object_num(self):
        env = AntThermalRegulationEnv(n_food=10)
        env.reset()

        n_food = 0
        for obj in env.objects:
            if obj[3] is ObjectClass.FOOD:
                n_food += 1

        assert n_food == 10

    def test_default_object_num(self):
        env = AntThermalRegulationEnv()
        env.reset()

        n_food = 0
        for obj in env.objects:
            if obj[3] is ObjectClass.FOOD:
                n_food += 1

        assert n_food == 6

    def test_object_num_small(self):
        env = AntSmallThermalRegulationEnv(n_food=10)
        env.reset()

        n_food = 0
        for obj in env.objects:
            if obj[3] is ObjectClass.FOOD:
                n_food += 1

        assert n_food == 10

    def test_dt(self):
        env = AntThermalRegulationEnv()
        env.reset()

        assert env.dt == 0.01 * 5

    @pytest.mark.parametrize("seed,expected,expected_val",
                             [
                                 (0, 0, 0.7765381891168206),
                                 (1, 1, 0.6882236302311676),
                                 (2, 2, 0.7746062316006963)
                             ])
    def test_multi_modal_dims(self, seed, expected, expected_val):
        env = AntThermalRegulationEnv()
        seed_value = env.seed(seed)

        assert seed_value == [expected]

        obs = env.reset()

        assert obs[0] == approx(expected_val)

    def test_max_time_steps(self):
        env = AntThermalRegulationEnv()
        env._max_episode_steps = 10

        num_of_decisions = 0
        env.reset()

        while True:
            a = env.action_space.sample()
            num_of_decisions += 1

            _, _, done, _ = env.step(a)

            if done:
                break

        assert num_of_decisions == 10

    def test_max_time_steps_init(self):
        env = AntThermalRegulationEnv(max_episode_steps=42)

        num_of_decisions = 0
        env.reset()

        while True:
            a = env.action_space.sample()
            num_of_decisions += 1

            _, _, done, _ = env.step(a)

            if done:
                break

        assert num_of_decisions == 42

    def test_scale_temp(self):
        env = AntThermalRegulationEnv()

        assert env.scale_temperature(42) == approx(1.0)
        assert env.scale_temperature(34) == approx(-1.0)
        assert env.scale_temperature(38) == approx(0.0)

    def test_decode_temp(self):
        env = AntThermalRegulationEnv()

        assert env.decode_temperature(1) == approx(42)
        assert env.decode_temperature(-1) == approx(34)
        assert env.decode_temperature(0.0) == approx(38)

    @pytest.mark.parametrize("temp",
                             [
                                 -0.3,
                                 0.3,
                                 -0.2,
                                 0.2,
                                 -0.1,
                                 0.1,
                                 -0.05,
                                 0.05,
                                 0,
                             ])
    def test_render_temp_env(self, temp):
        env = AntSmallThermalRegulationEnv(show_sensor_range=True, n_bins=20, sensor_range=16., visualize_temp=True)
        env.reset()
        env.thermal_model.reset(temp_init=env.decode_temperature(temp))
        for i in range(10):
            env.step(env.action_space.sample())
            env.render()
        env.close()

    def test_rgb(self):
        env = AntThermalRegulationEnv()
        env.reset()

        im = None
        for i in range(100):
            env.step(env.action_space.sample())
            im = env.render(mode='rgb_array', camera_id=0, width=32, height=32)

        import matplotlib.pyplot as plt
        plt.imshow(im)
        plt.savefig("test_im.png")

        assert im.shape == (32, 32, 3)

        env.close()

    def test_depth(self):
        env = AntThermalRegulationEnv()
        env.reset()

        with pytest.raises(AssertionError):
            env.render(mode='depth_array', camera_id=0, width=32, height=32)

    def test_default_climate(self):
        env = AntThermalRegulationEnv()
        env.reset()

        assert env.get_climate() is Climate.default

    @pytest.mark.parametrize("climate",
                             [
                                 Climate.default,
                                 Climate.cloudy,
                                 Climate.sunny,
                                 None,
                             ])
    def test_set_climate(self, climate):
        env = AntThermalRegulationEnv()
        prev_climate = env.get_climate()

        env.reset(climate=climate)

        if climate is None:
            assert env.get_climate() is prev_climate
        else:
            assert env.get_climate() is climate

    def test_random_climate(self):
        n = 5000.0
        env = AntThermalRegulationEnv(random_climate=True)

        climate_dict = {c: 0 for c in Climate}
        for i in range(int(n)):
            env.reset()
            climate_dict[env.get_climate()] += 1

        assert climate_dict[Climate.default] / n == approx(expected=0.0)
        assert climate_dict[Climate.sunny] / n == approx(expected=0.5, abs=0.1)
        assert climate_dict[Climate.cloudy] / n == approx(expected=0.5, abs=0.1)

    @pytest.mark.parametrize("climate",
                             [
                                 Climate.default,
                                 Climate.cloudy,
                                 Climate.sunny,
                             ])
    def test_climate_choice_is_stronger_than_random_climate(self, climate):
        env = AntThermalRegulationEnv(random_climate=True)

        for i in range(20):
            env.reset(climate=climate)
            assert env.get_climate() is climate

    @pytest.mark.parametrize("reward_setting,expected,param",
                             [
                                 ("homeostatic", -2, None),
                                 ("homeostatic_shaped", +2, None),
                                 ("one", 0, None),
                                 ("homeostatic_biased", -1.5, 0.5),
                             ])
    def test_modular_reward(self, reward_setting, expected, param):
        env = AntSmallThermalRegulationEnv(reward_setting=reward_setting, coef_main_rew=1.0, reward_bias=param)

        action = env.action_space.sample() * 0

        env.prev_interoception = np.array([1, 1])
        env.internal_state = {
            InteroClass.ENERGY: 0.0,
            InteroClass.TEMPERATURE: 0.0,
        }

        if reward_setting == "one":
            env.internal_state = {
                InteroClass.ENERGY: -0.99999,
                InteroClass.TEMPERATURE: -0.999999,
            }
            _, reward, done, info = env.step(action)

            assert done
            assert info["reward_module"] is None
        else:
            _, reward, done, info = env.step(action)
            rm = info["reward_module"]
            assert rm.shape == (3, )


    def test_intero_obs_position(self):
        env = AntSmallThermalRegulationEnv(internal_reset="random")

        for _ in range(10):
            env.reset()

            obs, _, _, info = env.step(env.action_space.sample())

            assert np.all(obs[-2:] == info["interoception"])


    def test_intero_dim(self):
        env = AntSmallThermalRegulationEnv(internal_reset="random")
        assert env.dim_intero == 2


    def test_update_n_food(self):
        env = AntSmallThermalRegulationEnv(internal_reset="random")

        env.reset()
        assert len(env.objects) == 6

        env.reset(n_food=2)
        assert len(env.objects) == 2

        env.reset(n_food=10)
        assert len(env.objects) == 10

        env.reset(n_food=0)
        assert len(env.objects) == 0
