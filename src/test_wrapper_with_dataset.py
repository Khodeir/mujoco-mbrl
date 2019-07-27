import time
import numpy as np
from src.mbrl.env_wrappers import EnvWrapper
from src.mbrl.data import Rollout, TransitionsDataset


def random_rollout(environment, num_steps):
    states, actions, observations, rewards = [], [], [], []
    for _ in range(num_steps):
        action = environment.sample_action()
        state, observation, reward = environment.step(action)
        states.append(state)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)

        _ = environment.reset()
    return Rollout(states=states, observations=observations, actions=actions[:-1], rewards=rewards[1:])


def pretty_print_transition(inps, targets):
    print("Inputs: \n \t state = {} \n \t action = {}".format(np.around(inps[0], decimals=3),
                                                              np.around(inps[2], decimals=3)))
    print("\t observations:")
    for k, v in inps[1].items():
        print("\t \t {} : {}".format(k, np.around(v, decimals=3)))
    print("Targets: \n \t state = {}, \n \t reward = {}".format(np.around(targets[1], decimals=3),
                                                                np.around(targets[0], decimals=3)))


env = EnvWrapper.load(env_name="reacher", task_name="easy")
episode_len = 500
num_episodes = 10
start = time.time()
rollouts = [random_rollout(environment=env, num_steps=episode_len) for _ in range(num_episodes)]
mid = time.time()
print("took {:.3f} s to to rollout env".format(mid - start))
dataset = TransitionsDataset(rollouts=rollouts, transitions_capacity=100000)
print("took {:.3f} s to build transition dataset".format(time.time() - mid))
print("Dataset contains {} transitions".format(len(dataset)))

unnormalised_dataset = TransitionsDataset(rollouts=rollouts, transitions_capacity=100000, normalise=False)

normalised_transition = dataset.get_transition(roll_idx=1, start_idx=30)
unnormalised_transition = unnormalised_dataset.get_transition(roll_idx=1, start_idx=30)

# Quick check on normalisation values
manual_norm_state = (unnormalised_transition[0][0][0] - dataset.statistics["states"]["mean"]) / \
                    dataset.statistics["states"]["std"]
assert np.allclose(normalised_transition[0][0][0], manual_norm_state)

manual_norm_action = (unnormalised_transition[0][0][2] - dataset.statistics["actions"]["mean"]) / \
                     dataset.statistics["actions"]["std"]
assert np.allclose(normalised_transition[0][0][2], manual_norm_action)

manual_norm_reward = (unnormalised_transition[1][0][0] - dataset.statistics["rewards"]["mean"]) / \
                     dataset.statistics["rewards"]["std"]
assert np.allclose(normalised_transition[1][0][0], manual_norm_reward)

print(" ++++++++++++++++++  Normalised Transition ++++++++++++++++++ ")
pretty_print_transition(normalised_transition[0][0], normalised_transition[1][0])

print("\n ++++++++++++++++++ Unnormalised Transition ++++++++++++++++++ ")
pretty_print_transition(unnormalised_transition[0][0], unnormalised_transition[1][0])


# Now checking multistep transitions
print("\n ++++++++++++++++++ Multistep Transition ++++++++++++++++++ ")
horizon = 5
dataset = TransitionsDataset(rollouts=rollouts, horizon=horizon)
multistep_transition = dataset.get_transition()
print("Dataset contains {} {}-step transitions from {} rollouts".format(len(dataset), horizon, dataset.num_rollouts))

for i, trans in enumerate(zip(multistep_transition[0], multistep_transition[1])):
    print(" ------- step = {} -------".format(i))
    pretty_print_transition(*trans)
