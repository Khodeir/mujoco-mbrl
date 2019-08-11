from src.mbrl.env_wrappers import EnvWrapper
import numpy as np
env = EnvWrapper.load('reacher', 'hard')

rewards = []
for i in range(20):
    rollout = env.get_rollout(
        num_steps=100,
        set_state=True,
    )
    rewards.extend(rollout.rewards[1:])
rollout_rewards = np.array(rewards)
print('no sampling', np.min(rollout_rewards), np.mean(rollout_rewards), np.max(rollout_rewards), (rollout_rewards > 0).sum(), len(rollout_rewards))

rewards = []
for i in range(20):
    initial_state = env.set_goal_state()
    rollout = env.get_rollout(
        num_steps=100,
        set_state=True,
        goal_state=initial_state,
        initial_state=initial_state
    )
    rewards.extend(rollout.rewards[1:])
rollout_rewards = np.array(rewards)
print('sampling', np.min(rollout_rewards), np.mean(rollout_rewards), np.max(rollout_rewards), (rollout_rewards > 0).sum(), len(rollout_rewards))
