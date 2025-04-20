from trp_env.envs.snake_trp_env import SnakeTwoResourceEnv, SnakeSmallTwoResourceEnv


class TestEnv:

    def test_instance(self):
        env = SnakeTwoResourceEnv(
            ego_obs=True,
            no_contact=False,
            sparse=False
        )

    def test_reset_env(self):
        env = SnakeTwoResourceEnv()
        env.reset()

    def test_run_env(self):
        env = SnakeTwoResourceEnv()
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

    def test_dim(self):
        env = SnakeTwoResourceEnv()
        obs = env.reset()

        assert len(env.observation_space.high) == 12 + 10 + 10 + 2  # 17 + 10 + 10 if non-ego-centric observation
        assert len(env.action_space.high) == 4
        assert len(obs) == 12 + 10 + 10 + 2 # 17 + 10 + 10 if non-ego-centric observation
        assert len(env.action_space.sample()) == 4
