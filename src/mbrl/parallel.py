import torch.multiprocessing as multiprocessing# this needs to be at the top so it's called by all child processes
multiprocessing.set_sharing_strategy('file_system')
import torch
import os
from collections import namedtuple

from src.mbrl.env_wrappers import EnvWrapper
from src.mbrl.data import TransitionsDataset
from functools import partial
from operator import itemgetter
from typing import Dict


CollectionRequest = namedtuple(
    "CollectionRequest",
    ["env_name", "task_name", "flat_obs", "get_rollouts_kwargs", "index"],
)

DEFAULT_NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 2))
def get_rollouts_parallel(
    env_name, task_name, flat_obs, num_rollouts, get_rollouts_kwargs, num_workers=DEFAULT_NUM_WORKERS
):
    with multiprocessing.Pool(processes=num_workers) as pool:
        processes = [
            pool.apply_async(
                collect_from_env,
                (CollectionRequest(
                    env_name=env_name,
                    task_name=task_name,
                    flat_obs=flat_obs,
                    get_rollouts_kwargs=get_rollouts_kwargs,
                    index=i
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
    if 'mp4path' in request.get_rollouts_kwargs:
        request.get_rollouts_kwargs['mp4path'] += '_{}'.format(request.index)
        rollout = env.record_rollout(**request.get_rollouts_kwargs)
    else:
        rollout = env.get_rollout(**request.get_rollouts_kwargs)
    return rollout
