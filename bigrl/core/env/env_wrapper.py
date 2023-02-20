import gym

class RewardRecordWrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._cumulative_rewards = 0.

    def reset(self, **kwargs):
        self._cumulative_rewards = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._cumulative_rewards += reward
        info['cumulative_rewards'] = self._cumulative_rewards
        return observation, reward, done, info