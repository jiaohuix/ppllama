# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
'''
cmd: python -m paddle.distributed.launch  scripts/example.py --mp 1 --ckpt_dir ckpt/7B/ --tokenizer_path  ckpt/tokenizer.model
'''
from typing import Tuple
import os
import sys
import paddle
import fire
import time
import json
import random
import numpy as np
from pathlib import Path
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from collections import OrderedDict
from ppllama import ModelArgs, Transformer, Tokenizer, LLaMA


def set_random_seed(seed, rank_id):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed + rank_id)

def change_dtype(checkpoint, dtype="float32"):
    for k, v in checkpoint.items():
        checkpoint[k] = v.astype(dtype)

def setup_model_parallel(mp_degree: int = 1) -> Tuple[int, int]:
    strategy = fleet.DistributedStrategy()

    # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/model_parallel_cn.html#erdongtaitushiyongfangfa
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


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    paddle.set_device("cpu")
    # paddle.set_default_dtype("float32")

    checkpoints = sorted(Path(ckpt_dir).glob("*.pdparams"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading ckpt from disk...")
    #PosixPath' object has no attribute 'tell' ; issue: https://github.com/PaddlePaddle/Paddle/issues/34614#issuecomment-1074719350
    checkpoint = paddle.load(str(ckpt_path))
    # Variable [ vocab_parallel_embedding_0.w_0 ] need tensor with dtype paddle.float32  but load tensor with dtype paddle.float16
    change_dtype(checkpoint)

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=4, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    print("Loading model...")
    model = Transformer(model_args)
    print("Loading state dict to model...")
    model.set_dict(checkpoint)
    del checkpoint
    paddle.set_device("gpu")
    print("Moving model from cpu to cuda...")
    model.to("gpu")

    # model parallel
    model = fleet.distributed_model(model)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(ckpt_dir: str, tokenizer_path: str, mp: int = 1, temperature: float = 0.8, top_p: float = 0.95):
    local_rank, world_size = setup_model_parallel(mp_degree=mp)
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)
    # prompts = ["The capital of Germany is the city of",
    #            "Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]

    # prompt_bsz <= max_batch_size=1   ; https://github.com/facebookresearch/llama/issues/42#issuecomment-1451321954
    prompts = ["What is baidu's  PaddlePaddle?", "The capital of Germany is the city of",  "I believe the meaning of life is","write a story about a grain of sand as it watches millions of years go by"]

    start_time = time.time()
    print("Inferencing...")
    results = generator.generate(prompts, max_gen_len=256, temperature=temperature, top_p=top_p)

    for idx,result in enumerate(results):
        print(f"[{idx}] Promt is: \n {prompts[idx]} \n")
        print(f"result is:\n {result}")
        print("\n==================================\n")

    print(f"Inferencing {len(prompts)} prompts takes {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    fire.Fire(main)