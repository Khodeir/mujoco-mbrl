from src.mbrl.agents import iLQRAgent, MPCAgent, GDAgent
from tqdm import tqdm
from dm_control import suite

import os
import sys
import argparse
from tensorboardX import SummaryWriter
import torch
import numpy as np

# from src.mbrl.utils import *
# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train a model using iLQR MPC')
# Experiment parameters
parser.add_argument('--environment', default='cheetah-run', help='Environment to train the agent on')
parser.add_argument("--num-trials", default=5, type=int, help="Number of trials for this experiment")
parser.add_argument("--agg-iters", default=1, type=int, help="Aggregation iterations")
parser.add_argument("--traj-per-agg", default=5, type=int, help="Number of rollouts per aggregation iteration")
parser.add_argument("--traj-length", default=500, type=int, help="Length of rollouts")
parser.add_argument("--test-open-loop", default=False, type=bool, help="Test open loop state prediction errors")
# Planner parameters
parser.add_argument('--planner', default='ilqr', help='The type of planner to use from [ilqr, gd, mpc]')
parser.add_argument("--H", default=10, type=int, help="Planning horizon of the controller")
parser.add_argument("--K", default=1000, type=int, help="Num rollouts of the random MPC controller")
parser.add_argument("--numIter", default=70, type=int, help="Max number of iterations in trajectory optimiser")
parser.add_argument("--stop", default=2e-3, type=float, help="Stop criterion for trajectory optimiser")
# Model Parameters
parser.add_argument('--model', default='linear', help='The type of model to use from [linear, nn]')
parser.add_argument('--num-hidden', default=10, type=int, help='The number of hidden units in NN model')
parser.add_argument("--multistep", default=1, type=int, help="Number of forward steps in training")
parser.add_argument("--num-epochs", default=200, type=int, help="Number of epochs to train model")
parser.add_argument('--noise_level', default=None, type=float, help='The noise level on the model state predictions')

args = parser.parse_args()
# Main loops
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
assets_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets'))
# Some info about the experiment
env_name = args.environment.split('-')[0]
task_name = args.environment.split('-')[1]

if args.model == 'linear':
    model_name = '-Linear'
elif args.model == 'nn':
    model_name = '-NN_' + str(args.num_hidden)
elif args.model == 'rnn':
    model_name = '-RNN_' + str(args.num_hidden)

# Using swimmer3 for now - (swimmer6 is 16+2)
state_dims = {'point_mass': 4, 'reacher': 4, 'cheetah': 18-1+2, 'manipulator': 22+7, 'humanoid': 55+5,
              'swimmer': 10+2, 'walker': 18-1+3, 'hopper': 14-1+4}

for iteration in range(args.num_trials):
    state_dim = state_dims[env_name]
    goal = torch.zeros(state_dim, dtype=torch.float)
    # -------------------- create agent + experiment name for logs ----------------------------------------
    if args.planner == 'ilqr':
        experiment_name = env_name + model_name + '-iLQR-H=' + str(args.H) + '-epochs=' + str(args.num_epochs) +\
                          '-ms=' + str(args.multistep) + '-trial=' + str(iteration)
        if args.noise_level:
            experiment_name += '-noise=' +str(args.noise_level)
        writer = SummaryWriter(log_dir=os.path.join('logs', experiment_name))

        agent = iLQRAgent(goalState=goal, env_name=env_name, task_name=task_name, model=args.model,
                          num_hidden=args.num_hidden, noise=args.noise_level, multistep=args.multistep, H=args.H,
                          LQRIter=args.numIter, LQRStop=args.stop, traj_length=args.traj_length, writer=writer)
        
    elif args.planner == 'gd':
        experiment_name = env_name + model_name + '-GD-H=' + str(args.H) + '-epochs=' + str(args.num_epochs) + '-ms='\
                          + str(args.multistep) + '-trial=' + str(iteration)
        if args.noise_level:
            experiment_name += '-noise=' + str(args.noise_level)
        writer = SummaryWriter(log_dir=os.path.join('logs', experiment_name))

        agent = GDAgent(goalState=goal, env_name=env_name, task_name=task_name, model=args.model,
                        num_hidden=args.num_hidden, noise=args.noise_level, multistep=args.multistep, H=args.H,
                        GDIter=args.numIter, GDStop=args.stop, traj_length=args.traj_length, writer=writer)

    elif args.planner == 'mpc':
        experiment_name = env_name + model_name + '-MPC-H=' + str(args.H) + '-K=' + str(args.K) + '-epochs=' + \
                          str(args.num_epochs) + '-ms=' + str(args.multistep) + '-trial=' + str(iteration)
        if args.noise_level:
            experiment_name += '-noise=' + str(args.noise_level)
        writer = SummaryWriter(log_dir=os.path.join('logs', experiment_name))

        agent = MPCAgent(goalState=goal, env_name=env_name, task_name=task_name, model=args.model,
                         num_hidden=args.num_hidden, noise=args.noise_level, multistep=args.multistep,
                         traj_length=args.traj_length, H=args.H, K=args.K, writer=writer)
    
    for agg_iter in range(args.agg_iters):
        print('Aggregation iteration number: ', agg_iter)

        if args.model == 'rnn':
            agent.train_rnn(args.num_epochs, agg_iter)
        else:
            agent.train(args.num_epochs, agg_iter)  # Training also normalises the data
        # Unnormalising it because I'm going to be appending unnormalised states in the generated trajectories
        agent.D.unnormalise_dataset()
        # ------------------------------ TEST OPEN LOOP PREDICTIONS ----------------------------------------
        if args.test_open_loop:
            print('Testing model open-loop prediction errors...')
            num_trials = 1000
            max_horizon = 31
            for h in tqdm(range(1, max_horizon, 2)):
                # total_loss = 0
                losses = []
                for _ in range(num_trials):
                    true, predicted = agent.test_H_step_pred(H=h)
                    t = torch.tensor(true, dtype=torch.float)
                    p = torch.tensor(predicted, dtype=torch.float)
                    # loss = torch.mean((t[1: h+1] - p[1: h+1])**2) #### Is this the quantity that I want?
                    loss = torch.mean((t[h: h+1] - p[h: h+1])**2)
                    losses.append(loss)

                losses = np.array(losses)
                print('Open-loop loss for ', h, ' steps forward = ', np.mean(losses))
                writer.add_scalar('open-loop average loss', np.mean(losses), h)
                writer.add_scalar('open-loop loss std', np.std(losses), h)
            print('Done testing open-loop prediction errors. Moving to generating trajectories with planner...')
        # ----------------------------------------------------------------------------------------------------
        for traj in range(args.traj_per_agg):
            timestep = agent.env.reset()
            # If it's humanoid manually initialise it to standing
            if env_name == 'humanoid':
                initial_state = sample_humanoid_state()
                with agent.env.physics.reset_context():
                    agent.env.physics.set_state(initial_state)
            # If it's walker manually initialise it to standing
            if env_name == 'walker':
                initial_state = sample_walker_state()
                with agent.env.physics.reset_context():
                    agent.env.physics.set_state(initial_state)
            # If it's hopper manually initialise it to standing
            if env_name == 'hopper':
                initial_state = sample_hopper_state()
                with agent.env.physics.reset_context():
                    agent.env.physics.set_state(initial_state)
            # If it's swimmer manually initialise it to standing
            if (env_name == 'swimmer'):
                initial_state = agent.env.physics.state()
                initial_state[2] = np.random.uniform(-3.14, 3.14)
                with agent.env.physics.reset_context():
                    agent.env.physics.set_state(initial_state)

            if (args.planner == 'ilqr') or (args.planner == 'gd'):
                print('Initialising a nominal trajectory...')
                agent.XList, agent.UList = agent.initialise_trajectory()

            total_reward = 0
            total_state_cost = 0
            total_action_cost = 0
            recorder = Recorder(experiment_name + '-' + str(agg_iter), traj)
            init_state = agent.get_state()
            new_traj = [init_state]

            new_goal_interval = {'point_mass': 150, 'reacher': 100, 'cheetah': 1000, 'manipulator': 150,
                                 'swimmer': 1000, 'humanoid': 1000, 'walker': 1000, 'hopper': 1000}
            for t in tqdm(range(args.traj_length)):
                # -------------------- set a goal state for point-mass ------------------------------
                if env_name == 'point_mass':
                    if t % new_goal_interval[env_name] == 0:
                        target = np.random.uniform(-0.25, 0.25, 3)
                        target[-1] = 0.01
                        agent.env.physics.named.model.geom_pos['target'] = target
                        goalState = torch.zeros(state_dim, dtype=torch.float)
                        goalState[0] = target[0]
                        goalState[1] = target[1]
                # -------------------- set a goal state for reacher ------------------------------
                if env_name == 'reacher':
                    if t % new_goal_interval[env_name] == 0:
                        state = agent.env.physics.get_state()
                        with agent.env.physics.reset_context():
                            agent.env.physics.set_state(state)

                            goalState = torch.zeros(state_dim, dtype=torch.float)
                            goalState[0] = np.random.uniform(low=-np.pi, high=np.pi)
                            goalState[1] = np.random.uniform(low=-2.8, high=2.8)  # Avoid infeasible goals
                            goalState[-1] = 0
                            goalState[-2] = 0

                            a = 0.12*np.cos(goalState[1])
                            b = 0.12*np.sin(goalState[1])
                            theta = goalState[0] + np.arctan(b/(0.12 + a))
                            mag = np.sqrt((0.12 + a)**2 + b**2)
                            target_x = mag*np.cos(theta)
                            target_y = mag*np.sin(theta)
                            agent.env.physics.named.model.geom_pos['target', 'x'] = target_x
                            agent.env.physics.named.model.geom_pos['target', 'y'] = target_y
                # -------------------- set a goal state for cheetah ------------------------------
                if env_name == 'cheetah':
                    if t % new_goal_interval[env_name] == 0:
                        goalState = torch.zeros(state_dim, dtype=torch.float)
                        goalState[-2] = 2  # Target Speed
                        goalState[-1] = 0.4  # Target torso height
                # ------------------------------ set a goal state for swimmer ------------------------------
                if env_name == 'swimmer':
                    target = agent.env.physics.named.data.geom_xpos['target'][:2]
                    if t % new_goal_interval[env_name] == 0:
                        goalState = torch.zeros(state_dim, dtype=torch.float)
                        goalState[0] = target[0]
                        goalState[1] = target[1]
                # ------------------------------------------------------------------------------------------
                if env_name == 'manipulator':
                    if t % new_goal_interval[env_name] == 0:
                        goalState = torch.zeros(state_dim, dtype=torch.float)
                        target_ball = agent.env.physics.body_location('target_ball')[:2]
                        # Want the ball to be over the target
                        goalState[8] = target_ball[0]
                        goalState[9] = target_ball[1]
                        # Want the gripper location to be at the target
                        goalState[-7] = target_ball[0]
                        goalState[-6] = target_ball[1]
                        goalState[-5:] = .5  # Contact sensors
                # --------------------------------------------------------------------------------
                if env_name == 'humanoid':
                    if t % new_goal_interval[env_name] == 0:
                        goalState = torch.zeros(state_dim, dtype=torch.float)
                        # Want the gripper location to be at the target
                        # goalState[-4] = 1.69 ### head height
                        # goalState[-3] = 1.0 ### torso height
                        # goalState[-2:] = 0.0 ## torso velocities

                        # All the penalties and velocities are 0s
                # --------------------------------------------------------------------------------
                if env_name == 'walker':
                    if t % new_goal_interval[env_name] == 0:
                        goalState = torch.zeros(state_dim, dtype=torch.float)
                        goalState[-3] = 1.0  # target torso_upright
                        goalState[-2] = 1.3  # target torso_height
                        goalState[-1] = 3.0  # 0.  ## target speed
                # --------------------------------------------------------------------------------
                if env_name == 'hopper':
                    if t % new_goal_interval[env_name] == 0:
                        goalState = torch.zeros(state_dim, dtype=torch.float)
                        goalState[-2] = 0.9 ## target torso_height
                        goalState[-1] = 1.0 ##  ## target speed
                # --------------------------------------------------------------------------------
                agent.set_goal(agent.normalise_state(goalState))

                recorder.record_frame(agent.env.physics.render(camera_id=0), t)
                if (args.planner == 'ilqr') or (args.planner == 'gd'):
                    if t % 25 == 0:
                       agent.XList, agent.UList = agent.initialise_trajectory()

                action = agent.choose_action()
                new_traj.append(action)

                timestep = agent.env.step(action)
                new_state, reward = agent.get_state(), timestep.reward
                normalised_new_state = agent.get_state(normalise=True)
                if reward is None:
                    reward = 0
                new_traj.append(reward)
                new_traj.append(new_state)

                if t % 20 == 1:
                    print("++++++++++++++++++++++++++++++++ Time Step ", t, " ++++++++++++++++++++++++++++++++")
                    print("Current state (normalised) = ", normalised_new_state)
                    print("Current action = ", action)
                    print("Unnormalised goal state = ", goalState)
                    print("Normalised goal state = ", agent.goalState)
                    print("Current state cost = ",
                          agent.state_cost(torch.tensor(normalised_new_state, dtype=torch.float)))
                    print("Current action cost = ", agent.action_cost(torch.tensor(action, dtype=torch.float)))
                    if (args.planner == 'ilqr') or (args.planner == 'gd'):
                        print("Current planned state trajectory starts at = ", agent.XList[0])
                        print("Current planned state trajectory ends at = ", agent.XList[-1])
                        print("The last state in plan has state cost = ", agent.state_cost(agent.XList[-1]))
                        print("Current planned actions = ", agent.UList)

                if reward:
                    total_reward += reward
                total_state_cost += agent.state_cost(torch.tensor(normalised_new_state, dtype=torch.float))
                total_action_cost += agent.action_cost(torch.tensor(action, dtype=torch.float))

            agent.D_RL.push_rollout(new_traj)
            
            print('Trajectory done. Total reward: {}'.format(total_reward))
            writer.add_scalar('total_reward/{}'.format(agg_iter), total_reward, traj)
            writer.add_scalar('total_state_cost/{}'.format(agg_iter), total_state_cost, traj)
            writer.add_scalar('total_action_cost/{}'.format(agg_iter), total_action_cost, traj)
            writer.add_scalar('total_cost/{}'.format(agg_iter), total_state_cost + total_action_cost, traj)

            recorder.make_movie() 


