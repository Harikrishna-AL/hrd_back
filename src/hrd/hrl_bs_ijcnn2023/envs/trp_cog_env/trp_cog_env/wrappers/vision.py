from collections import deque
from typing import Tuple

import numpy as np
import gym
import torch
from gym.spaces import Box

from trp_env.envs.two_resource_env import BIG


class TRPVisionEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env, im_size: Tuple[int, int], n_frame: int, mode: str):
        """

        :param env: environment to learn
        :param im_size: image size (height, width)
        :param n_frame:  number of frames to stack
        :param mode: vision input mode. rgb_array or rgbd_array.
        """
        super().__init__(env)

        self.n_frame_stack = n_frame
        self.im_size = im_size

        assert mode in {"rgb_array", "rgbd_array"}
        self.mode = mode

        self.frame_stack = deque(maxlen=n_frame)

        self.obs_dims = env.multi_modal_dims

        dummy_obs = self.reset()
        ub = BIG * np.ones(dummy_obs.shape, dtype=np.float32)
        self.obs_space = gym.spaces.Box(ub * -1, ub)

    def reset(self, **kwargs):
        self.frame_stack.clear()
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    @property
    def observation_space(self):
        return self.obs_space

    def observation(self, observation):
        # scale into (-1, 1)
        vision = 2. * (self.env.render(mode=self.mode, height=self.im_size[0], width=self.im_size[1], camera_id=0).astype(np.float32) / 255. - 0.5)

        if len(self.frame_stack) < self.n_frame_stack:
            for i in range(self.n_frame_stack):
                self.frame_stack.append(vision.flatten())
        else:
            self.frame_stack.append(vision.flatten())

        # observation of low-dim two-resource is [proprioception(27 dim), exteroception (40 dim, default), interoception (2dim)]
        proprioception = observation[:self.obs_dims[0]]
        interoception = observation[-self.obs_dims[2]:]

        fullvec = np.concatenate([proprioception, interoception, np.concatenate(self.frame_stack)])
        return fullvec

    def decode_vision(self, im):
        return 0.5 * im + 0.5


def to_multi_modal(obs_tensor,
                   im_size: Tuple[int, int],
                   n_frame: int,
                   n_channel: int,
                   mode="ant"):
    """
    function to convert the flatten multimodal observation to invididual modality
    :param obs_tensor: observation obtained from data: Size = [n_batch, channel, height, width]
    :param im_size: size of the image (height, width)
    :param n_frame: number of stacks of observation
    :param n_channel: number of channel of vision (rgb:3, rgbd:4)
    :param mode: environment mode. "ant", "humanoid", or "goldfish". default: "ant"
    :return:
    """

    if mode == "ant":
        dim_prop = 27
    elif mode == "goldfish":
        dim_prop = 24
    elif mode == "humanoid":
        dim_prop = 66
    else:
        raise ValueError("mode variable should be trp or goldfish.")

    sizes = (dim_prop, 2, np.prod(im_size) * n_channel * n_frame)

    # TODO: remove torch dependency
    input_prop, input_intero, input_vision = torch.split(obs_tensor, split_size_or_sections=sizes, dim=1)

    input_vision = torch.split(input_vision, split_size_or_sections=[np.prod(im_size) * n_channel] * n_frame, dim=1)

    input_vision = [im.reshape(-1, im_size[0], im_size[1], n_channel) for im in input_vision]

    input_vision = torch.cat(input_vision, dim=3)

    input_vision = input_vision.view(-1, im_size[0], im_size[1], n_channel * n_frame).permute(0, 3, 1, 2)

    return input_prop, input_intero, input_vision
