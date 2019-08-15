if __name__ == '__main__':
    import torch.multiprocessing as multiprocessing
    multiprocessing.set_start_method('forkserver')
    from src.mbrl.experiment import *
    from src.mbrl.parallel import get_rollouts_parallel

    config = dict(
        exp_dir="par",
        agent=Agent.GoalStateAgent,
        environment=Environment("reacher_hard"),
        planner=Planner.RandomShooting,
        model=Model.NeuralNet,
        num_train_iterations=0,
        optimizer=Optimizer.Adam,
        horizon=20,
        rollout_length=15,
        num_rollouts_per_iteration=1
    )




    agent = main(config)



    policy = agent.policy


    rollouts = get_rollouts_parallel("reacher", "hard", True, 100, policy.get_action, 6, 3)

    print(len(rollouts))

