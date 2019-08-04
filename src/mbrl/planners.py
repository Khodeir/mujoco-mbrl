import torch
import torch.nn
import numpy as np
import copy

# Types:
from typing import Callable, Optional, List, Tuple
from torch import Tensor

Trajectory = Tuple[List[torch.Tensor], List[torch.Tensor]]
ScalarTorchFunc = Callable[[torch.Tensor], float]
TensorTorchFunc = Callable[[torch.Tensor], torch.Tensor]
TorchFunc = Callable[[], torch.Tensor]

class ModelPlanner:
    @staticmethod
    def plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        state_cost: ScalarTorchFunc,
        action_cost: ScalarTorchFunc,
        sample_action: TorchFunc,
        horizon: int,
        initial_trajectory: Optional[Trajectory] = None,
        **kwargs
    ) -> Trajectory:
        raise NotImplementedError


class GradientDescentPlanner(ModelPlanner):
    defaults = dict(num_iterations=100, stop_condition=0.001)

    @staticmethod
    def plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        state_cost: ScalarTorchFunc,
        action_cost: ScalarTorchFunc,
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
            state_cost=state_cost,
            action_cost=action_cost,
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
        state_cost: ScalarTorchFunc,
        action_cost: ScalarTorchFunc,
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
            state_cost=state_cost,
            action_cost=action_cost,
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
        state_cost: ScalarTorchFunc,
        action_cost: ScalarTorchFunc,
        initial_trajectory: Trajectory,
        num_iterations: int,
        stop_condition: float,
        horizon: int,
    ) -> Trajectory:
        """
        Iteratively apply gradient descent to the trajectory w.r.t. actions selected in initial sequence
        """
        _, action_list = initial_trajectory

        for action in action_list:
            action.requires_grad = True
        initial_state.requires_grad = False

        traj_optimizer = torch.optim.Adam(action_list, lr=0.01)

        for _ in range(num_iterations):
            traj_optimizer.zero_grad()
            # run model forward from init state, accumulate losses
            state_list = [initial_state.unsqueeze(dim=0)]  # a list of torch tensors
            # Loop forwards, accumulate costs
            for i in range(horizon):
                next_state = model(state_list[-1], action_list[i])
                state_list.append(next_state)
            loss = torch.sum(state_cost(torch.stack(state_list[1:])) + action_cost(torch.stack(action_list)))
            # Backprop losses to actions
            loss.backward(retain_graph=True)
            # Update actions
            old_actions = copy.deepcopy(action_list)
            # -------------------------------- NEW --------------------------------------------
            # if iteration == 0:
            #     grad_norms = torch.tensor([u.grad.norm() for u in self.UList])
            #     print('First grad norms of actions = ', grad_norms)
            # ---------------------------------------------------------------------------------
            traj_optimizer.step()
            new_actions = action_list[:]

            change_amount = 0.0
            for j in range(horizon):
                change_amount += torch.mean(
                    torch.abs(new_actions[j] - old_actions[j])
                ).detach().numpy()
            change_amount /= horizon

            if change_amount < stop_condition:
                break

        return state_list, [a.detach() for a in action_list]


class RandomShootingPlanner(ModelPlanner):
    defaults = dict(num_trajectories=1000)

    @staticmethod
    def plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        state_cost: ScalarTorchFunc,
        action_cost: ScalarTorchFunc,
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
            state_cost,
            action_cost,
            sample_action,
            horizon,
            initial_trajectory,
            num_trajectories,
        )

    @staticmethod
    def _plan(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        state_cost: ScalarTorchFunc,
        action_cost: ScalarTorchFunc,
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
            state_cost=state_cost,
            action_cost=action_cost,
        )
        chosen_trajectory_idx = np.argmin(trajectory_costs)
        trajectory = trajectories[chosen_trajectory_idx]

        return trajectory

    @staticmethod
    def _generate_trajectories(
        initial_state: torch.Tensor,
        model: TensorTorchFunc,
        state_cost: ScalarTorchFunc,
        action_cost: ScalarTorchFunc,
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
                states = state_list[i * num_trajectories : (i + 1) * num_trajectories]
            actions = action_list[i * num_trajectories : (i + 1) * num_trajectories]
            # infer with model
            state_list[i * num_trajectories : (i + 1) * num_trajectories] = model(states, actions)
        costs = (state_cost(state_list) + action_cost(action_list)).view(horizon, num_trajectories).sum(0).detach().numpy()
        trajectories = []
        state_list = state_list.view((horizon, num_trajectories, -1))
        action_list = action_list.view((horizon, num_trajectories, -1))
        for i in range(num_trajectories):
            trajectories.append((state_list[:, i], action_list[:, i]))
        return trajectories, costs

