import torch
import torch.nn
import numpy as np
import copy
from tqdm import tqdm
from src.mbrl.data import Data
from src.mbrl.models import Model, LinearModel, SmoothAbsLoss, CoshLoss

from torch.utils.data import DataLoader
from src.mbrl.env_wrappers import EnvWrapper


class Agent:

    def __init__(self, env_name, task_name, goal_state, model, num_hidden, multistep=1, noise=None, traj_length=100,
                 H=10, K=1000, writer=None):


        self.env = EnvWrapper.load(env_name, task_name, visualize_reward=True)
        self.env_name = env_name
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_spec().shape[0]
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

        self.set_goal(goal_state)
        self.action_cost = CoshLoss()

        # TODO: Put this in some config file
        init_data = {'point_mass': (100, 250), 'reacher': (100, 250), 'cheetah': (500, 200),
                     'manipulator': (500, 150), 'swimmer': (200, 300), 'humanoid': (1000, 40), 'walker': (1500, 50),
                     'hopper': (100, 200)}  # Walker was 3000 x 25
        self.D_rand = self.get_random_data(*init_data[self.env_name])
        self.D_RL = Data()
        self.writer = writer

    def set_goal(self, goal_state):
        self.goal_state = goal_state
        weights = torch.from_numpy(self.env.get_goal_weights())
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
            action = self.env.sample_action()
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
        # TODO: Something about the fact that this returns a torch tensor or a numpy array depending on normalize
        state = self.env.get_state()
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
                initial_state = self.env.sample_state()
                with self.env.physics.reset_context():
                    self.env.physics.set_state(initial_state)
                s0 = self.get_state(normalise=False)
                trajectory = [s0, ]
                for i in range(traj_length):
                    action = self.env.sample_action()
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
            actions = [self.env.sample_action() for _ in range(roll_length)]
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


    def normalise_state(self, state):
        state = torch.tensor(state, dtype=torch.float)
        return (state - self.D.state_mean) / self.D.state_std

    def unnormalise_state(self, state):
        return state * self.D.state_std + self.D.state_mean


