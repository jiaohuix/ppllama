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
from ppllama.utlis import setup_model_parallel,load_model

def main(ckpt_dir: str, tokenizer_path: str, prompt:str = "The capital of Germany is the city of",
         mp: int = 1, temperature: float = 0.8, top_p: float = 0.95):
    paddle.set_device("cpu")
    local_rank, world_size = setup_model_parallel(mp_degree=mp)
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    model, generator = load_model(ckpt_dir, tokenizer_path, local_rank, world_size)
    prompts = [prompt]
    results = generator.generate(prompts, max_gen_len=256, temperature=temperature, top_p=top_p)

    for idx,result in enumerate(results):
        print(f"Promt [{idx}]  is: \n {prompts[idx]} \n")
        print(f"result is:\n {result}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)