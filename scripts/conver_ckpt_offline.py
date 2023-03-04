import os
import torch
import sys
import paddle
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

def print_state(state):
    for k, v in state.items():
        print(k, v.shape)

def convert_pytorch_checkpoint_to_numpy(
        torch_ckpt_path="model.pth",
        outfolder="."
):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    # torch_to_paddle = {}
    do_not_transpose = ["tok_embeddings.weight"]

    pytorch_state_dict = torch.load(torch_ckpt_path, map_location="cpu")
    for idx, (k, v) in tqdm(enumerate(pytorch_state_dict.items())):
        outname = os.path.join(outfolder, f"{idx}-{k}.npy")
        is_transpose = False
        if k.endswith(".weight") and (v.ndim == 2) and (k not in do_not_transpose):
            is_transpose = True
            v = v.transpose(0, 1)

        print(f"Converting {idx + 1} key: {k} | is_transpose {is_transpose}")
        np.save(outname, v.data.numpy())
    del pytorch_state_dict


def save_paddle_checkpoint_from_numpy(torch_ckpt_path, outfolder):
    paddle_ckpt_path = os.path.join(outfolder, os.path.basename(torch_ckpt_path).replace("pth", "pdparams"))
    np_ckpt_names = list(sorted(os.listdir(outfolder), key=lambda x: int(x.split("-")[0])))
    paddle_state_dict = OrderedDict()
    for name in np_ckpt_names:
        np_file = os.path.join(outfolder, name)
        key = name.split("-")[1].replace(".npy", "")
        paddle_state_dict[key] = np.load(np_file).astype("float32")
        os.remove(np_file)

    paddle.save(paddle_state_dict, paddle_ckpt_path)


if __name__ == '__main__':
    assert len(sys.argv) >=3 , f"usage: python {sys.argv[0]} <torch_ckpt_path>(xx.pth)  <np_ckpt_dir> <merge(y/n)>(optional)"
    torch_ckpt_path = sys.argv[1]
    np_ckpt_dir = sys.argv[2]
    merge = True
    if len(sys.argv) == 4 and sys.argv[3] == "n":
        merge = False
    convert_pytorch_checkpoint_to_numpy(torch_ckpt_path, outfolder)
    if merge:
        save_paddle_checkpoint_from_numpy(torch_ckpt_path, outfolder)