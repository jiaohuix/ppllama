import torch
import sys
import paddle
from collections import OrderedDict
from tqdm import tqdm

def print_state(state):
    for k,v in state.items():
        print(k, v.shape)

def convert_pytorch_checkpoint_to_paddle(
    torch_ckpt_path="model.pth",
    paddle_dump_path="model.pdparams",
):
    # torch_to_paddle = {}
    do_not_transpose = ["tok_embeddings.weight"]
    print("Loading ckpt...")
    pytorch_state_dict = torch.load(torch_ckpt_path,map_location="cpu")
    print("Converting torch ckpt to paddle...")
    print_state(pytorch_state_dict)
    paddle_state_dict = OrderedDict()
    for k,v in pytorch_state_dict.items():
        is_transpose = False
        if k.endswith(".weight") and (v.ndim == 2) and (k not in do_not_transpose):
            is_transpose = True
            v = v.transpose(0, 1)

        print(f"Converting: key: {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()

    del pytorch_state_dict
    paddle.save(paddle_state_dict, paddle_dump_path)

if __name__ == '__main__':
    assert len(sys.argv) == 2, f"usage: python {sys.argv[0]} <torch_ckpt_path>(xx.pth)"
    torch_ckpt_path = sys.argv[1]
    convert_pytorch_checkpoint_to_paddle(torch_ckpt_path,file.replace("pth","pdparams"))