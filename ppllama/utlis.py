from typing import Tuple
import os
import gc
import sys
import paddle
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
from tqdm import tqdm

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

# def change_dtype(checkpoint, dtype="float32"):
#     for k, v in checkpoint.items():
#         checkpoint[k] = v.astype(dtype)

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
    set_random_seed(1)

    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    return local_rank, world_size

def load_pp_weights(ckpt_path, model):
    '''support two format of weights:
        ckpt_path = xx.pdparams
        1. pp weight file:  xx.pdparams                 [50G]
        2. np weight directory:  xx.pdparams/idx-key.npy [slow but memory efficient 31G]
        '''
    if os.path.isdir(ckpt_path):
        np_ckpt_names = list(sorted(os.listdir(ckpt_path), key=lambda x: int(x.split("-")[0])))
        print("Loading shard state dict to model...")
        for name in tqdm(np_ckpt_names):
            np_file = os.path.join(ckpt_path, name)
            key = name.split("-")[1].replace(".npy", "")
            model.set_dict({key:np.load(np_file).astype("float32")})
            # if key in model.state_dict().keys():
            #     paddle.assign(
            #         x = paddle.to_tensor(np.load(np_file).astype("float32")),
            #         output = model.state_dict()[key]
            #     )
        gc.collect()
    else:
        print("Loading state from disk...")
        #PosixPath' object has no attribute 'tell' ; issue: https://github.com/PaddlePaddle/Paddle/issues/34614#issuecomment-1074719350
        checkpoint = paddle.load(str(ckpt_path))
        print("Loading state dict to model...")
        model.set_dict(checkpoint)
        del checkpoint
        gc.collect()
    paddle.set_device("gpu")
    print("Moving model from cpu to cuda...")
    model.to("gpu")


def load_model(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    paddle.set_device("cpu")
    paddle.set_default_dtype("float32")
    checkpoints = sorted(Path(ckpt_dir).glob("*.pdparams"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    # 1 load model
    print("Loading model...")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=4, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args)
    print(len(model.state_dict()))

    # 2 load ckpt
    load_pp_weights(ckpt_path, model)

    # 3 model parallel
    model = fleet.distributed_model(model)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, generator
