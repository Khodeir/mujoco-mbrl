import torch
import torch.nn
import numpy as np

# Types:
from typing import Callable, Optional, List, Tuple

Trajectory = Tuple[List[torch.Tensor], List[torch.Tensor]]
ScalarTorchFunc = Callable[[torch.Tensor, torch.Tensor], float]
TensorTorchFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TorchFunc = Callable[[int], torch.Tensor]


class ModelPlanner:
    @staticmethod
    def plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        cost: ScalarTorchFunc,
        sample_action: TorchFunc,
        horizon: int,
        initial_trajectory: Optional[Trajectory] = None,
        **kwargs
    ) -> Trajectory:
        raise NotImplementedError


class GradientDescentPlanner(ModelPlanner):
    defaults = dict(num_iterations=40, stop_condition=0.002)

    @staticmethod
    def plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        cost: ScalarTorchFunc,
        sample_action: TorchFunc,
        horizon: int,
        initial_trajectory: Optional[Trajectory] = None,
        **kwargs
    ) -> Trajectory:
        num_iterations = kwargs.get(
            "num_iterations", GradientDescentPlanner.defaults["num_iterations"]
        )
        stop_condition = kwargs.get(
            "stop_condition", GradientDescentPlanner.defaults["stop_condition"]
        )
        return GradientDescentPlanner._plan(
            initial_state=initial_state,
            model=model,
            cost=cost,
            sample_action=sample_action,
            horizon=horizon,
            initial_trajectory=initial_trajectory,
            num_iterations=num_iterations,
            stop_condition=stop_condition,
        )

    @staticmethod
    def _plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        cost: ScalarTorchFunc,
        sample_action: TorchFunc,
        horizon: int,
        initial_trajectory: Optional[Trajectory],
        num_iterations: int,
        stop_condition: float,
    ) -> Trajectory:
        if initial_trajectory is None:
            initial_trajectory = GradientDescentPlanner._initialise_trajectory(
                initial_state=initial_state,
                model=model,
                sample_action=sample_action,
                horizon=horizon,
            )

        trajectory = GradientDescentPlanner._optimize_trajectory(
            initial_state=initial_state,
            cost=cost,
            model=model,
            initial_trajectory=initial_trajectory,
            num_iterations=num_iterations,
            stop_condition=stop_condition,
            horizon=horizon,
        )
        return trajectory

    @staticmethod
    def _initialise_trajectory(initial_state, model, sample_action, horizon):
        """
        Initialises some nominal sequence to start the trajectory optimisation with. Both sequences are normalised.
        """
        state_list = [initial_state.unsqueeze(dim=0)]  # a list of torch tensors
        action_list = list(sample_action(batch_size=horizon).split(1, dim=0))
        for i in range(horizon):
            next_state = model(state_list[-1], action_list[i])
            state_list.append(next_state)

        return state_list, action_list

    @staticmethod
    def _optimize_trajectory(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        cost: ScalarTorchFunc,
        initial_trajectory: Trajectory,
        num_iterations: int,
        stop_condition: float,
        horizon: int,
    ) -> Trajectory:
        """
        Iteratively apply gradient descent to the trajectory w.r.t. actions selected in initial sequence
        """
        _, action_list = initial_trajectory
        states_tensor = torch.zeros((horizon + 1, initial_state.shape[-1]))
        states_tensor[0] = initial_state
        actions_tensor = torch.cat(action_list, dim=0)
        actions_tensor.requires_grad = True
        traj_optimizer = torch.optim.Adam([actions_tensor], lr=0.1)

        for _ in range(num_iterations):
            traj_optimizer.zero_grad()            
            # Loop forwards, accumulate costs
            for i in range(horizon):
                states_tensor[i+1] = model(states_tensor[i:i+1], actions_tensor[i:i+1])

            loss = torch.sum(cost(states_tensor[1:], actions_tensor))
            loss.backward(retain_graph=True)
            old_actions_tensor = actions_tensor.clone().detach()
            # Update actions
            traj_optimizer.step()

            change_amount = torch.mean(torch.abs(old_actions_tensor - actions_tensor)).detach().numpy()
            if change_amount < stop_condition:
                break

        return [s.detach() for s in states_tensor.split(1, 0)], [a.detach() for a in actions_tensor.split(1, 0)]


class RandomShootingPlanner(ModelPlanner):
    defaults = dict(num_trajectories=1000)

    @staticmethod
    def plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        cost: ScalarTorchFunc,
        sample_action: TorchFunc,
        horizon: int,
        initial_trajectory: Optional[Trajectory] = None,
        **kwargs
    ) -> Trajectory:
        num_trajectories = kwargs.get(
            "num_trajectories", RandomShootingPlanner.defaults["num_trajectories"]
        )
        return RandomShootingPlanner._plan(
            initial_state,
            model,
            cost,
            sample_action,
            horizon,
            initial_trajectory,
            num_trajectories,
        )

    @staticmethod
    def _plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        cost: ScalarTorchFunc,
        sample_action: TorchFunc,
        horizon: int,
        initial_trajectory: Optional[Trajectory],
        num_trajectories: int,
    ):
        trajectories, trajectory_costs = RandomShootingPlanner._generate_trajectories(
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            horizon=horizon,
            sample_action=sample_action,
            model=model,
            cost=cost,
        )
        chosen_trajectory_idx = np.argmin(trajectory_costs)
        trajectory = trajectories[chosen_trajectory_idx]

        return trajectory

    @staticmethod
    def _generate_trajectories(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        cost: ScalarTorchFunc,
        sample_action: TorchFunc,
        horizon: int,
        num_trajectories: int,
    ) -> Tuple[List[Trajectory], np.ndarray]:

        state_list = torch.zeros((num_trajectories * horizon, initial_state.shape[0]))
        action_list = sample_action(batch_size=num_trajectories * horizon)

        for i in range(horizon):
            if i == 0:
                states = initial_state.unsqueeze(dim=0).repeat_interleave(num_trajectories, dim=0)
            else:
                states = state_list[(i - 1) * num_trajectories : i * num_trajectories]
            actions = action_list[i * num_trajectories : (i + 1) * num_trajectories]
            # infer with model
            state_list[i * num_trajectories : (i + 1) * num_trajectories] = model(states, actions)
        costs = cost(state_list, action_list).view(horizon, num_trajectories).sum(0).detach().numpy()
        trajectories = []
        state_list = state_list.view((horizon, num_trajectories, -1))
        action_list = action_list.view((horizon, num_trajectories, -1))
        for i in range(num_trajectories):
            trajectories.append((state_list[:, i], action_list[:, i]))
        return trajectories, costs
