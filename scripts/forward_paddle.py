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
from ppllama.utlis import setup_model_parallel, load_model




def run_forward(ckpt_dir: str,tokenizer_path:str, local_rank: int, world_size: int) -> LLaMA:

    model, generator = load_model(ckpt_dir, tokenizer_path, local_rank, world_size, use_gpu=False)
    #### Pseudo forward ####
    reprod_logger = ReprodLogger()

    bsz, seq = 4, 64
    tokens = np.random.randint(0, 1024, [bsz, seq])
    tokens = paddle.to_tensor(tokens)
    reprod_logger.add("input_tokens", tokens.detach().numpy())

    logits = model(tokens, start_pos=0)
    reprod_logger.add("logits", logits.detach().numpy())
    reprod_logger.save("forward_paddle.npy")


def main(ckpt_dir: str, tokenizer_path: str):
    paddle.set_device("cpu")
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    run_forward(ckpt_dir,tokenizer_path, local_rank, world_size)


if __name__ == "__main__":
    fire.Fire(main)