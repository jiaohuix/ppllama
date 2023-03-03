# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import torch.nn as nn

from pathlib import Path
import numpy as np
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    seed = 1
    np.random.seed(1)
    torch.manual_seed(seed)
    return local_rank, world_size


def reset_paramaters(model):
    for m in model.modules():
        if isinstance(m, ParallelEmbedding) or isinstance(m, RowParallelLinear) or isinstance(m, ColumnParallelLinear):
            m.weight.data.normal_(0, 0.02)


def main(ckpt_dir: str):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=32, **params)
    model_args.vocab_size = 1024
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    reset_paramaters(model)
    torch.set_default_tensor_type(torch.FloatTensor)
    outfile = os.path.join(ckpt_dir, "consolidated.00.pth")
    torch.save(model.state_dict(), outfile)
    print(f"Pseudo weights have been saved to path: {outfile}")


if __name__ == "__main__":
    fire.Fire(main)