import torch
import torch.nn
import numpy as np
import copy
from tqdm import tqdm
from src.mbrl.data import Data
from src.mbrl.models import Model, LinearModel, SmoothAbsLoss, CoshLoss

from torch.utils.data import DataLoader

from dm_control import suite
from src.mbrl.utils import sample_humanoid_state, sample_hopper_state, sample_walker_state


class Agent:

    def __init__(self, env_name, task_name, goalState, model, num_hidden, multistep=1, noise=None, traj_length=100,
                 H=10, K=1000, writer=None):

        state_dims = {'point_mass': 4, 'reacher': 4, 'cheetah': 18 - 1 + 2, 'manipulator': 22 + 7, 'humanoid': 55 + 5,
                      'swimmer': 10 + 2, 'walker': 18 - 1 + 3, 'hopper': 14 - 1 + 4}

        self.env = suite.load(env_name, task_name, visualize_reward=True)
        self.env_name = env_name
        self.action_spec = self.env.action_spec()
        self.state_dim = state_dims[env_name]
        self.action_dim = self.action_spec.shape[0]
        self.multistep = multistep
        self.num_hidden = num_hidden
        self.model_type = model
        self.H = H
        self.K = K
        # ---- Also ugly ----
        self.RNN = False

        if model == 'linear':
            self.model = LinearModel(self.state_dim, self.action_dim, noise=noise)
        elif model == 'nn':
            self.model = Model(state_dim=self.state_dim, action_dim=self.action_dim, hidden_units=num_hidden,
                               noise=noise)
        elif model == 'rnn':
            self.model = RNNModel(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=num_hidden)
            self.RNN = True

        self.criterion = torch.nn.MSELoss()
        l2_penalty = 0  # 0.001
        learn_rate = 0.001

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=l2_penalty)

        self.set_goal(goalState)
        self.action_cost = CoshLoss()

        init_data = {'point_mass': (100, 250), 'reacher': (100, 250), 'cheetah': (500, 200),
                     'manipulator': (500, 150), 'swimmer': (200, 300), 'humanoid': (1000, 40), 'walker': (1500, 50),
                     'hopper': (100, 200)}  # Walker was 3000 x 25
        self.D_rand = self.get_random_data(*init_data[self.env_name])
        self.D_RL = Data()
        self.writer = writer

    def set_goal(self, goal_state):
        self.goalState = goal_state
        state_penalty = 1.0
        weights = torch.zeros(self.state_dim)
        # --------------------------------------------------------------------------------
        if self.env_name == 'point_mass':
            weights[0:2] = 10 * state_penalty
            weights[2:] = state_penalty / 4.  # Penalties on the velocities act as dampers

        elif self.env_name == 'reacher':
            weights[0:2] = state_penalty
            weights[2:] = state_penalty / 20.  # Penalties on the velocities act as dampers

        elif self.env_name == 'cheetah':
            weights[17] = state_penalty
            weights[18] = state_penalty / 2.

        elif self.env_name == 'swimmer':
            weights[0:1] = 10 * state_penalty  # Distance to target
            # weights[18] = state_penalty/2.
            weights[5:-2] = state_penalty  # velocities
            # weights[8:-2] = state_penalty/4. ## velocities

        elif self.env_name == 'manipulator':
            weights = torch.zeros(self.state_dim)
            weights[8:10] = 10 * state_penalty
            weights[10:21] = state_penalty / 4
            weights[-7:-5] = 10 * state_penalty
            weights[-5:] = state_penalty / 20

        elif self.env_name == 'humanoid':
            weights = torch.zeros(self.state_dim)
            weights[-5:] = 10 * state_penalty

        elif self.env_name == 'walker':
            weights = torch.zeros(self.state_dim)
            weights[-3:] = state_penalty

        elif self.env_name == 'hopper':
            weights = torch.zeros(self.state_dim)
            weights[-2] = state_penalty / 2.
            weights[-1] = state_penalty
        # --------------------------------------------------------------------------------
        self.state_cost = SmoothAbsLoss(weights=weights, goal_state=goal_state)

    def test_H_step_pred(self, H=10):
        """
        Randomly generate a rollout, and use the model to predict H-step open loop predictions
        """
        _ = self.env.reset()
        s0 = self.get_state(normalise=True)

        true_trajectory = [s0.numpy(), ]
        predicted_trajectory = [s0.numpy(), ]

        for i in range(H):
            action = self.sample_single_action()
            _ = self.env.step(action)
            next_state = self.get_state(normalise=True)

            s = torch.tensor(predicted_trajectory[-1], dtype=torch.float)
            a = torch.tensor(action, dtype=torch.float)
            joint_state = torch.cat([s, a])
            predicted_next_state = self.model(joint_state)

            true_trajectory.append(next_state.numpy())
            predicted_trajectory.append(predicted_next_state.detach().numpy())

        return true_trajectory, predicted_trajectory

    def get_state(self, normalise=False):
        # Reacher and Point-Mass
        if (self.env_name == 'point_mass') or (self.env_name == 'reacher'):
            state = self.env.physics.state()
        elif self.env_name == 'manipulator':
            state = self.env.physics.state()
            state = np.append(state, self.env.physics.named.data.site_xpos['grasp', 'x'])  # gripper position
            state = np.append(state, self.env.physics.named.data.site_xpos['grasp', 'z'])  # gripper position
            state = np.append(state, self.env.physics.touch())  # Sensors. 5 dimensions
        elif self.env_name == 'cheetah':
            state = self.env.physics.state()[1:]
            state = np.append(state, self.env.physics.speed())  # Add horizontal speed
            state = np.append(state, self.env.physics.named.data.subtree_com['torso'][2])  # Add torso height
        elif self.env_name == 'swimmer':
            # Assume swimmer6 for now
            state = self.env.physics.state()  # 16-2 dims
            state = np.append(state, self.env.physics.named.data.xmat['head'][:2])  # Head orientation (2)
            # state = np.append(state, self.env.physics.joints()) ##All internal joint angles (K-1) (5)
            # state = np.append(state, self.env.physics.body_velocities()) ## All local velocities (3*K) (18)
        elif self.env_name == 'humanoid':
            state = self.env.physics.state()  # 55 (pure state)
            # state = np.append(state, self.env.physics.head_height()) ## +1 head height (target should be 1.6)
            # state = np.append(state, self.env.physics.torso_upright()) ## +1 torso projected height (target should be 1.0)
            com_pos = self.env.physics.center_of_mass_position()
            rfoot = self.env.physics.named.data.xpos['right_foot']
            lfoot = self.env.physics.named.data.xpos['left_foot']
            ave_foot = (rfoot + lfoot) / 2.0
            above_feet = ave_foot + np.array([0., 0., 1.3])
            torso = self.env.physics.named.data.xpos['torso']

            first_penalty = np.linalg.norm(com_pos[:2] - ave_foot[:2])  # First described by Tassa
            second_penalty = np.linalg.norm(com_pos[:2] - torso[:2])  # Second term described by Tassa
            third_penalty = np.linalg.norm(torso[1:] - above_feet[1:])  # Third term

            state = np.append(state, first_penalty)  # +1
            state = np.append(state, second_penalty)  # +1
            state = np.append(state, third_penalty)  # +1
            state = np.append(state, self.env.physics.center_of_mass_velocity()[:2])  # +2 velocity should be 0

        elif self.env_name == 'walker':
            state = self.env.physics.state()[1:]
            state = np.append(state, self.env.physics.torso_upright())  # Add torso upright
            state = np.append(state, self.env.physics.torso_height())  # Add torso height
            state = np.append(state, self.env.physics.horizontal_velocity())  # Add horizontal speed
        elif self.env_name == 'hopper':
            state = self.env.physics.state()[1:]  # Removed horizontal position
            state = np.append(state, self.env.physics.touch())  # Add touch sensors in feet
            state = np.append(state, self.env.physics.height())  # Add torso height
            state = np.append(state, self.env.physics.speed())  # Add horizontal speed

        return self.normalise_state(state) if normalise else state

    def set_linear_model(self, a, b, noise_rate=0.0):
        """
        Manually set linear model parameters for testing simple environments
        """
        print('Setting linear model parameters manually...')
        weight_noise = torch.randn_like(self.model.linear1.weight.data) * noise_rate
        bias_noise = torch.randn_like(self.model.linear1.bias.data) * 0  # *noise_rate

        self.model.linear1.weight.data = torch.tensor(np.concatenate((a, b), axis=1), dtype=torch.float) + weight_noise
        self.model.linear1.bias.data = torch.zeros(self.state_dim, dtype=torch.float) + bias_noise

    def get_random_data(self, num_rolls=1000, traj_length=30):
        """
        Would like to experiment with this - how about uniformly sampling the state space?
        """
        D = Data()
        print('Generating D_rand')
        print('Each rollout has length = ', traj_length)
        for _ in tqdm(range(num_rolls)):
            try:
                # Initial timestep
                _ = self.env.reset()
                # ----------- Humanoid needs a different setup - make it start standing
                if self.env_name == 'humanoid':
                    initial_state = sample_humanoid_state()
                    with self.env.physics.reset_context():
                        self.env.physics.set_state(initial_state)
                # ----------- Walker needs a different setup - make it start standing
                if self.env_name == 'walker':
                    initial_state = sample_walker_state()
                    with self.env.physics.reset_context():
                        self.env.physics.set_state(initial_state)
                # ----------- Hopper needs a different setup - make it start standing
                if self.env_name == 'hopper':
                    initial_state = sample_hopper_state()
                    with self.env.physics.reset_context():
                        self.env.physics.set_state(initial_state)
                # ----------- Swimmer needs a different setup - randomise heading direction
                if self.env_name == 'swimmer':
                    initial_state = self.env.physics.state()
                    initial_state[2] = np.random.uniform(-3.14, 3.14)
                    with self.env.physics.reset_context():
                        self.env.physics.set_state(initial_state)
                # ------------------------------------------------------------------
                s0 = self.get_state(normalise=False)
                trajectory = [s0, ]
                for i in range(traj_length):
                    action = self.sample_single_action()
                    trajectory.append(action)

                    timestep = self.env.step(action)
                    trajectory.append(timestep.reward)

                    next_state = self.get_state(normalise=False)
                    trajectory.append(next_state)

                    if timestep.last():
                        break
                # D.pushTrajectory(trajectory)
                D.push_multi_step_trajectory(trajectory, self.multistep)
            except:
                pass  # Nothing special needed for initialisation
        print('Generated {} samples'.format(len(D)))
        return D

    def aggregate_data(self):
        self.D = self.D_RL if len(self.D_RL) > 0 else self.D_rand

    def multistep_train(self, num_rollouts, iteration):
        """
        Use stochastic gradient descent on lots of length H rollouts to compute multistep loss and optimise
        model w.r.t. that loss instead of the single step one
        """
        self.aggregate_data()
        # ----------------- I'm only actually doing this to get statistics to normalise with -----------
        if len(self.D_RL) > 0:
            print('Now using D_RL...')
            self.D = self.D_RL
        else:
            print('Now using D_rand...')
            self.D = self.D_rand
            print('D = D_rand contains ', len(self.D), ' transitions')
        self.D.normalise_dataset()
        # ---------------------------------------------------------------------------------------------------
        roll_length = 200
        for rollout in tqdm(range(num_rollouts)):
            # ---------------------- Generate a length 'roll_length' rollout  ----------------------
            # 1) Randomly choose an initial state s0 and H random actions
            _ = self.env.reset()
            s0 = torch.tensor(self.get_state(normalise=True), dtype=torch.float)
            actions = [self.sample_single_action() for _ in range(roll_length)]
            # 2) Use the environment to generate a length H rollout from s0 using random actions
            targets = [s0]
            for a in actions:
                _ = self.env.step(a)
                targets.append(torch.tensor(self.get_state(normalise=True), dtype=torch.float))

            # ----------------------------------------------------------------------------------------
            # Slide over the entire generated trajectory in length H windows
            for i in range(roll_length - self.H - 1):
                state = targets[i]
                loss = 0.0
                # 3) Use the model to generate a length H rollout with same actions from s0, and compute losses
                for j in range(self.H):
                    action = actions[i + j]
                    joint_state = torch.cat([state, torch.tensor(action, dtype=torch.float)])
                    next_state = self.model(joint_state)
                    loss += self.criterion(next_state, targets[i + j + 1])
                    state = next_state
                # 4) Update model parameters
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

            if rollout % 50 == 0:
                if self.writer is not None:
                    self.writer.add_scalar('loss/state/{}'.format(iteration), loss / self.H, rollout)

    def train_rnn(self, num_epochs, iteration):
        self.aggregate_data()
        if len(self.D_RL) > 0:
            self.D = self.D_RL
            print('D = D_RL contains ', len(self.D), ' transitions')
            idx = np.random.randint(0, len(self.D))
            print('Random sample from D_RL = ', self.D[idx])
        else:
            self.D = self.D_rand
            print('D = D_rand contains ', len(self.D), ' transitions')
            idx = np.random.randint(0, len(self.D))
            print('Random sample from D_rand = ', self.D[idx])

        self.D.normalise_dataset()
        print('Same random sample from D after normalising = ', self.D[idx])

        batch = 512
        # ----------- Ugly, but need dataset to have length that's multiple of batch  -----------
        self.D.transitions = self.D.transitions[: -(len(self.D.transitions) % batch)]
        # ---------------------------------------------------------------------------------------------
        train_data = DataLoader(self.D, batch_size=batch, shuffle=True)
        for epoch in tqdm(range(num_epochs)):
            for trans in train_data:
                # ----------- Put training data in right format for RNNModel -----------
                initial_states = trans[0]
                actions = torch.tensor([], dtype=torch.float)
                targets = torch.tensor([], dtype=torch.float)
                for j in range(0, len(trans) - 3, 3):
                    a = trans[j + 1]
                    actions = torch.cat((actions, a))
                    next_state = trans[j + 3]
                    targets = torch.cat((targets, next_state))

                initial_states = initial_states.reshape((1, batch, self.state_dim))
                actions = actions.reshape((batch, self.multistep, self.action_dim))
                targets = targets.reshape((batch, self.multistep, self.state_dim))
                # -----------------------------------------------------------------------
                predictions = self.model(initial_states, actions)

                loss = self.criterion(predictions, targets)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

            if self.writer is not None:
                loss /= self.multistep
                self.writer.add_scalar('loss/state/{}'.format(iteration), loss, epoch)

    def train(self, num_epochs, iteration):
        self.aggregate_data()
        if len(self.D_RL) > 0:
            self.D = self.D_RL
            print('D = D_RL contains ', len(self.D), ' transitions')
            idx = np.random.randint(0, len(self.D))
            print('Random sample from D_RL = ', self.D[idx])
        else:
            self.D = self.D_rand
            print('D = D_rand contains ', len(self.D), ' transitions')
            idx = np.random.randint(0, len(self.D))
            print('Random sample from D_rand = ', self.D[idx])

        self.D.normalise_dataset()
        print('Same random sample from D after normalising = ', self.D[idx])

        batch = 512
        train_data = DataLoader(self.D, batch_size=batch, shuffle=True)

        for epoch in tqdm(range(num_epochs)):
            # if epoch % 50 == 49:
            #     print('saving model...')
            #     model_name = self.env_name + '-ms=' + str(self.multistep)
            #     if self.model_type == 'linear':
            #         model_name += '-Linear-epoch=' + str(epoch) + '.pt'
            #     elif self.model_type == 'nn':
            #         model_name += '-NN_' + str(self.num_hidden) + '-epoch=' + str(epoch) + '.pt'
            #     elif self.model_type == 'rnn':
            #         model_name += '-RNN_' + str(self.num_hidden) + '-epoch=' + str(epoch) + '.pt'
            #     save_path = '/Users/omakhlouf/Documents/Oxford/Project/RLProject/saved_models/' + model_name
            #     state = {'epoch': epoch,
            #              'state_dict': self.model.state_dict(),
            #              'optimizer': self.optimizer.state_dict()}
            #     torch.save(state, save_path)

            for trans in train_data:
                loss = 0
                state = trans[0]
                for j in range(0, len(trans) - 3, 3):
                    # state = trans[j]
                    action = trans[j + 1]
                    next_state = trans[j + 3]

                    state_action = torch.cat([state, action], 1)
                    next_state_hat = self.model(state_action)
                    if j == 0:
                        single_loss = self.criterion(next_state_hat, next_state)

                    loss += self.criterion(next_state_hat, next_state)
                    # loss = self.criterion(next_state_hat, next_state) ### What if I only train on the last one?
                    state = next_state_hat  # Use predicted next state in next iteration

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

            if self.writer is not None:
                loss /= self.multistep
                self.writer.add_scalar('loss/state/{}'.format(iteration), loss, epoch)
                self.writer.add_scalar('single_step_loss/{}'.format(iteration), single_loss, epoch)

    def old_train(self, num_epochs, iteration):
        self.aggregate_data()
        if len(self.D_RL) > 0:
            print('Now using D_RL...')
            self.D = self.D_RL
            print('D = D_RL contains ', len(self.D), ' transitions')
            idx = np.random.randint(0, len(self.D))
            print('Random sample from D_RL = ', self.D[idx])
        else:
            print('Now using D_rand...')
            self.D = self.D_rand
            print('D = D_rand contains ', len(self.D), ' transitions')
            idx = np.random.randint(0, len(self.D))
            print('Random sample from D_rand = ', self.D[idx])

        self.D.normalise_dataset()
        print('Same random sample from D after normalising = ', self.D[idx])

        batch = 512
        train_data = DataLoader(self.D, batch_size=batch, shuffle=True)

        for epoch in tqdm(range(num_epochs)):
            running_loss = 0
            for i, (state, action, reward, next_state) in enumerate(train_data):
                state_action = torch.cat([state, action], 1)
                next_state_hat = self.model(state_action)
                loss = self.criterion(next_state_hat, next_state)
                running_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.writer is not None:
                running_loss /= (int(len(self.D) / batch) + 1)
                self.writer.add_scalar('loss/state/{}'.format(iteration), running_loss, epoch)

    def sample_single_action(self):
        minimum = max(self.action_spec.minimum[0], -3)  # Clipping because LQR task has INF bounds
        maximum = min(self.action_spec.maximum[0], 3)
        action = np.random.uniform(minimum, maximum, self.action_spec.shape[0])

        if self.env_name == 'humanoid':
            action = np.random.normal(0, 0.4, self.action_dim)
            action[3:-6] = 0.

        return action

    def sample_batch_action(self, n):
        action = np.random.uniform(self.action_spec.minimum, self.action_spec.maximum, (n, self.action_spec.shape[0]))
        return action

    def normalise_state(self, state):
        state = torch.tensor(state, dtype=torch.float)
        return (state - self.D.state_mean) / self.D.state_std

    def unnormalise_state(self, state):
        return state * self.D.state_std + self.D.state_mean


class GDAgent(Agent):
    """
    An Agent that uses a first order gradient descent planner
    """

    def __init__(self, gd_iter, gd_stop, **kwargs):
        super(GDAgent, self).__init__(**kwargs)
        self.gd_iter = gd_iter
        self.gd_stop = gd_stop
        self.x_list, self.u_list = None, None

    def initialise_trajectory(self):
        """
        Initialises some nominal sequence to start the trajectory optimisation with. Both sequences are normalised.
        """
        state = torch.tensor(self.get_state(normalise=True), dtype=torch.float)
        x_list = [state]  # a list of torch tensors
        u_list = [torch.tensor(self.sample_single_action(), dtype=torch.float, requires_grad=True)
                  for _ in range(self.H)]

        for i in range(self.H):
            joint_state = torch.cat([x_list[-1], u_list[i]])
            next_state = self.model(joint_state)
            x_list.append(next_state)

        print('Initialised nominal trajectories...')
        return x_list, u_list

    def optimize_trajectory(self):
        """
        Iteratively apply gradient descent to the trajectory w.r.t. actions selected in initial sequence
        """
        if self.x_list is None:
            print('Initialising nominal trajectory...')
            self.x_list, self.u_list = self.initialise_trajectory()
        # ----------- Initialise next nominal trajectory with previous one shifted one timestep -----------
        else:
            new_x_list, new_u_list = self.x_list[1:], self.u_list[1:]
            # ----------- Get new random action, generate new last state ----------------------------------
            new_u_list.append(torch.tensor(self.sample_single_action(), dtype=torch.float, requires_grad=True))
            joint_state = torch.cat([new_x_list[-1], new_u_list[-1]])
            next_state = self.model(joint_state)
            new_x_list.append(next_state)
            # ---------------------------------------------------------------------------------------------
            self.x_list, self.u_list = new_x_list, new_u_list

        traj_optimizer = torch.optim.Adam(self.u_list, lr=0.01)

        for iteration in range(self.gd_iter):
            traj_optimizer.zero_grad()
            # run model forward from init state, accumulate losses
            s0 = torch.tensor(self.get_state(normalise=True), dtype=torch.float)
            x_list = [s0]  # a list of torch tensors
            loss = self.state_cost(x_list[0])
            # Loop forwards, accumulate costs
            for i in range(self.H):
                joint_state = torch.cat([x_list[-1], self.u_list[i]])
                next_state = self.model(joint_state)
                x_list.append(next_state)
                loss += (self.state_cost(next_state) + self.action_cost(self.u_list[i]))
            # Backprop losses to actions
            loss.backward(retain_graph=True)
            # Update actions
            old_actions = copy.deepcopy(self.u_list)
            # -------------------------------- NEW --------------------------------------------
            # if iteration == 0:
            #     grad_norms = torch.tensor([u.grad.norm() for u in self.UList])
            #     print('First grad norms of actions = ', grad_norms)
            # ---------------------------------------------------------------------------------
            traj_optimizer.step()
            new_actions = self.u_list[:]

            change_amount = 0
            for j in range(self.H):
                change_amount += torch.mean(torch.abs(new_actions[j] - old_actions[j]))
            change_amount /= self.H

            if change_amount < self.gd_stop:
                break

    def choose_action(self):
        self.optimize_trajectory()
        return self.u_list[0].detach().numpy()
