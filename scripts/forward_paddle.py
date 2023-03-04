# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import paddle
import fire
import time
import json
import numpy as np
from pathlib import Path
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from reprod_log import ReprodLogger
from ppllama import ModelArgs, Transformer, Tokenizer, LLaMA


def set_random_seed(seed, rank_id):
    np.random.seed(seed)
    paddle.seed(seed)


def setup_model_parallel(mp_degree: int = 1) -> Tuple[int, int]:
    strategy = fleet.DistributedStrategy()

    model_parallel_size = mp_degree
    data_parallel_size = 1
    strategy.hybrid_configs = {
        "dp_degree": data_parallel_size,
        "mp_degree": model_parallel_size,
        "pp_degree": 1
    }
    fleet.init(is_collective=True, strategy=strategy)

    hcg = fleet.get_hybrid_communicate_group()
    mp_id = hcg.get_model_parallel_rank()
    rank_id = dist.get_rank()

    # seed must be the same in all processes
    set_random_seed(1, rank_id)

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    return local_rank, world_size


def change_dtype(checkpoint, dtype="float32"):
    for k, v in checkpoint.items():
        checkpoint[k] = v.astype(dtype)


def run_forward(ckpt_dir: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pdparams"))
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = paddle.load(str(ckpt_path))
    change_dtype(checkpoint, dtype="float32")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=4, **params)
    paddle.set_default_dtype("float32")
    model = Transformer(model_args)
    model.eval()
    model.set_dict(checkpoint)

    #### Pseudo forward ####
    reprod_logger = ReprodLogger()

    bsz, seq = 4, 64
    tokens = np.random.randint(0, 1024, [bsz, seq])
    tokens = paddle.to_tensor(tokens)
    reprod_logger.add("input_tokens", tokens.detach().numpy())

    logits = model(tokens, start_pos=0)
    reprod_logger.add("logits", logits.detach().numpy())
    reprod_logger.save("forward_paddle.npy")


def main(ckpt_dir: str):
    paddle.set_device("cpu")
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    run_forward(ckpt_dir, local_rank, world_size)


if __name__ == "__main__":
    fire.Fire(main)