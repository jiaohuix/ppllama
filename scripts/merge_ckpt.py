import os
import sys
import paddle
from collections import OrderedDict
from tqdm import tqdm
import numpy as np


def save_paddle_checkpoint_from_numpy(torch_ckpt_path, np_ckpt_dir):
    paddle_ckpt_path = torch_ckpt_path.replace("pth", "pdparams")
    np_ckpt_names = list(sorted(os.listdir(np_ckpt_dir), key=lambda x: int(x.split("-")[0])))
    paddle_state_dict = OrderedDict()
    for name in np_ckpt_names:
        np_file = os.path.join(np_ckpt_dir, name)
        key = name.split("-")[1].replace(".npy", "")
        paddle_state_dict[key] = np.load(np_file).astype("float32")
        os.remove(np_file)

    paddle.save(paddle_state_dict, paddle_ckpt_path)


if __name__ == '__main__':
    assert len(sys.argv) == 3 , f"usage: python {sys.argv[0]} <torch_ckpt_path>  <np_ckpt_dir>"
    torch_ckpt_path = sys.argv[1]
    np_ckpt_dir = sys.argv[2]
    if not os.path.exists(np_ckpt_dir):
        os.makedirs(np_ckpt_dir)
    save_paddle_checkpoint_from_numpy(torch_ckpt_path, np_ckpt_dir)