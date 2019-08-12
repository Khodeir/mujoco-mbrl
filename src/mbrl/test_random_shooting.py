'''
This test will construct a toy example where the agent moves around in a 2d grid
where the cost function is distance from a particular grid location.
'''
import torch
world_size = 10
goal = torch.tensor(9, dtype=torch.float)
initial_state = torch.tensor([2], dtype=torch.float)
horizon=5

def model(states, actions):
    next_states = states + actions
    return torch.fmod((torch.fmod(next_states, world_size) + world_size), world_size)

def sample_action(batch_size):
    return torch.randint(low=-1, high=2, size=(batch_size,1), dtype=torch.float)

def cost(states, actions):
    return torch.abs(states - goal)


from src.mbrl.planners import RandomShootingPlanner


states, actions = RandomShootingPlanner.plan(initial_state, model, cost, sample_action, horizon, None, num_trajectories=1000)

print(states)
print(actions)


# trajs, costs = RandomShootingPlanner._generate_trajectories(initial_state, model, cost, sample_action, horizon, 3)
# for (traj, cost) in zip(trajs, costs):
#     states, actions = traj
#     print('states', states)

#     print('actions', actions)
#     break