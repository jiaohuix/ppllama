# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import numpy as np
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from reprod_log import ReprodLogger


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


def run_forward(ckpt_dir: str,tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
            world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=4, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.FloatTensor)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.eval()
    model = model.cpu()
    model.load_state_dict(checkpoint, strict=False)

    #### Pseudo forward ####
    reprod_logger = ReprodLogger()

    bsz, seq = 4, 64
    tokens = np.random.randint(0, 1024, [bsz, seq])
    tokens = torch.from_numpy(tokens).cpu()
    reprod_logger.add("input_tokens", tokens.detach().numpy())

    logits = model(tokens, start_pos=0)
    reprod_logger.add("logits", logits.detach().numpy())
    reprod_logger.save("forward_torch.npy")


def main(ckpt_dir: str,tokenizer_path: str):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    run_forward(ckpt_dir,tokenizer_path, local_rank, world_size)


if __name__ == "__main__":
    fire.Fire(main)