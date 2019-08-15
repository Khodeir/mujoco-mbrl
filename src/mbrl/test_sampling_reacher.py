from src.mbrl.env_wrappers import EnvWrapper
import numpy as np
env = EnvWrapper.load('reacher', 'hard')

normal_rollouts = [env.get_rollout(num_steps=100) for i in range(20)]
normal_rollout_rewards = np.array([r.rewards[1:] for r in normal_rollouts])
print('no sampling', np.min(normal_rollout_rewards), np.mean(normal_rollout_rewards), np.max(normal_rollout_rewards), (normal_rollout_rewards > 0).sum(), len(normal_rollout_rewards))

biased_rollouts = env.sample_rollouts_biased_rewards(num_rollouts=20, num_steps=100)
biased_rollout_rewards = np.array([r.rewards[1:] for r in biased_rollouts])
print('sampling', np.min(biased_rollout_rewards), np.mean(biased_rollout_rewards), np.max(biased_rollout_rewards), (biased_rollout_rewards > 0).sum(), len(biased_rollout_rewards))
assert np.mean(biased_rollout_rewards) > np.mean(normal_rollout_rewards)