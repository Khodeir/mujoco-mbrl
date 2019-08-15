import torch.multiprocessing as multiprocessing
import torch
from collections import namedtuple

from src.mbrl.env_wrappers import EnvWrapper
from src.mbrl.data import TransitionsDataset
from functools import partial
from operator import itemgetter
from typing import Dict


CollectionRequest = namedtuple(
    "CollectionRequest",
    ["num_steps", "env_name", "task_name", "flat_obs", "get_action"],
)  # ,'model', 'planner', 'stats', 'horizon'])


def get_rollouts_parallel(
    env_name, task_name, flat_obs, num_steps, get_action, num_rollouts, num_workers=2
):
    with multiprocessing.Pool(processes=num_workers) as pool:
        processes = [
            pool.apply_async(
                collect_from_env,
                (CollectionRequest(
                    num_steps=num_steps,
                    env_name=env_name,
                    task_name=task_name,
                    flat_obs=flat_obs,
                    get_action=get_action,
                ), )
            )
            for i in range(num_rollouts)
        ]
        rollouts = [p.get() for p in processes]
    return rollouts


def collect_from_env(request: CollectionRequest):
    env = EnvWrapper.load(
        env_name=request.env_name,
        task_name=request.task_name,
        flat_obs=request.flat_obs,
    )
    rollout = env.get_rollout(num_steps=request.num_steps, get_action=request.get_action)
    return rollout
