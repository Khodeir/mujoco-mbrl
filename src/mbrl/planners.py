import torch
import torch.nn
import numpy as np
import copy
from typing import Callable, Optional, List, Tuple

Trajectory = Tuple[List[np.ndarray], List[np.ndarray]]
# TODO: Standardize the types in plan's input/output. Pick either torch or numpy.
class ModelPlanner:
    @staticmethod
    def plan(
        initial_state: np.ndarray,
        model: Callable,
        state_cost: Callable,
        action_cost: Callable,
        get_action: Callable,
        horizon: int,
        initial_trajectory: Optional[Trajectory] = None,
        **kwargs
    ) -> Trajectory:
        raise NotImplementedError


class GradientDescentPlanner(ModelPlanner):
    defaults = dict(num_iterations=100, stop_condition=0.001)

    @staticmethod
    def plan(
        initial_state: np.ndarray,
        model: Callable,
        state_cost: Callable,
        action_cost: Callable,
        get_action: Callable,
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
            get_action=get_action,
            horizon=horizon,
            initial_trajectory=initial_trajectory,
            num_iterations=num_iterations,
            stop_condition=stop_condition,
        )

    @staticmethod
    def _plan(
        initial_state: np.ndarray,
        model: Callable,
        state_cost: Callable,
        action_cost: Callable,
        get_action: Callable,
        horizon: int,
        initial_trajectory: Optional[Trajectory],
        num_iterations: int,
        stop_condition: float,
    ) -> Trajectory:
        if initial_trajectory is None:
            initial_trajectory = GradientDescentPlanner._initialise_trajectory(
                initial_state=initial_state,
                model=model,
                get_action=get_action,
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
    def _initialise_trajectory(initial_state, model, get_action, horizon):
        """
        Initialises some nominal sequence to start the trajectory optimisation with. Both sequences are normalised.
        """
        state_list = [torch.tensor(initial_state, dtype=torch.float)]  # a list of torch tensors
        action_list = [
            torch.tensor(get_action(), dtype=torch.float, requires_grad=True)
            for _ in range(horizon)
        ]

        for i in range(horizon):
            joint_state = torch.cat([state_list[-1], action_list[i]])
            next_state = model(joint_state)
            state_list.append(next_state)

        return state_list, action_list

    @staticmethod
    def _optimize_trajectory(
        initial_state,
        model,
        state_cost,
        action_cost,
        initial_trajectory,
        num_iterations,
        stop_condition,
        horizon,
    ) -> Trajectory:
        """
        Iteratively apply gradient descent to the trajectory w.r.t. actions selected in initial sequence
        """
        # TODO This should really be the responsibility of some MPC controller
        # if self.state_list is None:
        #     print('Initialising nominal trajectory...')
        #     self.state_list, self.action_list = self.initialise_trajectory()
        # # ----------- Initialise next nominal trajectory with previous one shifted one timestep -----------
        # else:
        #     new_state_list, new_action_list = self.state_list[1:], self.action_list[1:]
        #     # ----------- Get new random action, generate new last state ----------------------------------
        #     new_action_list.append(torch.tensor(self.env.sample_action(), dtype=torch.float, requires_grad=True))
        #     joint_state = torch.cat([new_state_list[-1], new_action_list[-1]])
        #     next_state = self.model(joint_state)
        #     new_state_list.append(next_state)
        #     # ---------------------------------------------------------------------------------------------
        #     self.state_list, self.action_list = new_state_list, new_action_list
        state_list, action_list = initial_trajectory
        traj_optimizer = torch.optim.Adam(action_list, lr=0.01)

        for _ in range(num_iterations):
            traj_optimizer.zero_grad()
            # run model forward from init state, accumulate losses
            s0 = torch.tensor(initial_state, dtype=torch.float)
            state_list = [s0]  # a list of torch tensors
            loss = 0
            # Loop forwards, accumulate costs
            for i in range(horizon):
                joint_state = torch.cat([state_list[-1], action_list[i]])
                next_state = model(joint_state)
                state_list.append(next_state)
                loss += state_cost(next_state) + action_cost(action_list[i])
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

            change_amount = 0
            for j in range(horizon):
                change_amount += torch.mean(torch.abs(new_actions[j] - old_actions[j]))
            change_amount /= horizon

            if change_amount < stop_condition:
                break

        print("Initialised nominal trajectories...")
        return state_list, action_list


class RandomShootingPlanner(ModelPlanner):
    defaults = dict(num_trajectories=100)

    @staticmethod
    def plan(
        initial_state: np.ndarray,
        model: Callable,
        state_cost: Callable,
        action_cost: Callable,
        get_action: Callable,
        horizon: int,
        initial_trajectory: Optional[Trajectory] = None,
        **kwargs
    ) -> Trajectory:
        num_trajectories = kwargs.get('num_trajectories', RandomShootingPlanner.defaults['num_trajectories'])
        return RandomShootingPlanner._plan(
            initial_state,
            model,
            state_cost,
            action_cost,
            get_action,
            horizon,
            initial_trajectory,
            num_trajectories
        )

    @staticmethod
    def _plan(
        initial_state: np.ndarray,
        model: Callable,
        state_cost: Callable,
        action_cost: Callable,
        get_action: Callable,
        horizon: int,
        initial_trajectory: Optional[Trajectory],
        num_trajectories: int,
    ):

        trajectories, trajectory_costs = RandomShootingPlanner._generate_trajectories(
            initial_state=initial_state,
            num_trajectories=num_trajectories,
            horizon=horizon,
            get_action=get_action,
            model=model,
            state_cost=state_cost,
            action_cost=action_cost,
        )
        chosen_trajectory_idx = np.argmin(trajectory_costs)
        trajectory = [timestep[chosen_trajectory_idx] for timestep in trajectories]

        return trajectory

    @staticmethod
    def _generate_trajectories(
        initial_state: np.ndarray,
        model: Callable,
        state_cost: Callable,
        action_cost: Callable,
        get_action: Callable,
        horizon: int,
        num_trajectories: int,
    ):
        states = np.expand_dims(initial_state, 0).repeat(
            num_trajectories, 0
        )  # matrix size num_trajectories * state_dimensions, each row is the state
        trajectories = []
        costs = np.zeros(num_trajectories)
        for _ in range(horizon):
            # sample actions
            actions = [get_action() for _ in num_trajectories]
            # infer with model
            states = torch.tensor(states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.float)
            state_action = torch.cat([states, actions], 1)

            trajectories.append((states, actions))
            states = model(state_action)
            for i in range(num_trajectories):
                costs[i] += state_cost(states[i]) + action_cost(actions[i])

        return trajectories, costs

