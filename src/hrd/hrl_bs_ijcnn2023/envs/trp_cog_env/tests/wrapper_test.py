import pytest
import torch

from trp_env.wrappers.vision import TRPVisionEnvWrapper, to_multi_modal

from trp_env.envs.ant_trp_env import AntSmallTwoResourceEnv
from trp_env.envs.humanoid_trp_env import HumanoidSmallTwoResourceEnv


@pytest.mark.parametrize("expected, env_class, im_setting, n_frame",
                         [
                             (27 + 64 * 64 * 3 * 2 + 2, AntSmallTwoResourceEnv, "rgb_array", 2),
                             (27 + 64 * 64 * 4 * 4 + 2, AntSmallTwoResourceEnv, "rgbd_array", 4),
                             (66 + 64 * 64 * 4 * 5 + 2, HumanoidSmallTwoResourceEnv, "rgbd_array", 5),
                         ])
def test_vision_wrapper(expected, env_class, im_setting, n_frame):
    env = env_class()
    env = TRPVisionEnvWrapper(env,
                              im_size=(64, 64),
                              n_frame=n_frame,
                              mode=im_setting)

    obs = env.reset()

    assert obs.shape[0] == expected
    assert env.observation_space.shape == (expected,)


@pytest.mark.parametrize("shape_prop, shape_vision, shape_intero, n_channel, env_class, im_setting, n_frame",
                         [
                             ((1, 27), (1, 3 * 2, 64, 64), (1, 2), 3, AntSmallTwoResourceEnv, "rgb_array", 2),
                             ((1, 27), (1, 4 * 4, 64, 64), (1, 2), 4, AntSmallTwoResourceEnv, "rgbd_array", 4),
                             ((1, 66), (1, 4 * 5, 64, 64), (1, 2), 4, HumanoidSmallTwoResourceEnv, "rgbd_array", 5),
                         ])
def test_to_multi_model(shape_prop, shape_vision, shape_intero, n_channel, env_class, im_setting, n_frame):
    env = env_class()
    env = TRPVisionEnvWrapper(env,
                              im_size=(64, 64),
                              n_frame=n_frame,
                              mode=im_setting)

    obs = env.reset()

    mode = "ant" if env_class is AntSmallTwoResourceEnv else "humanoid"

    prop, intero, vision = to_multi_modal(torch.tensor([obs]),
                                          im_size=(64, 64),
                                          n_frame=n_frame,
                                          n_channel=n_channel,
                                          mode=mode)

    assert prop.shape == shape_prop
    assert intero.shape == shape_intero
    assert vision.shape == shape_vision
