import os
import sys
import paddle
from collections import OrderedDict
from tqdm import tqdm
import numpy as np



def fp_converter(in_ckpt, out_ckpt):

    pp_state = paddle.load(in_ckpt)
    for k,v in tqdm(pp_state.items()):
        pp_state[k] = pp_state[k].astype("float32")

    paddle.save(pp_state, out_ckpt)


if __name__ == '__main__':
    assert len(sys.argv) == 3, f"usage: python {sys.argv[0]} <in_ckpt> <out_ckpt>"
    in_ckpt = sys.argv[1]
    out_ckpt = sys.argv[2]
    fp_converter(in_ckpt, out_ckpt)
