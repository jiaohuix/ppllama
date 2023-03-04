# align

```shell
pip install reprod_log tqdm
git clone https://github.com/facebookresearch/llama.git
cd llama && pip install -r requirements.txt
pip install -e ./
```

Generate random weights for two layers:

```shell
torchrun --nproc_per_node 1 scripts/get_pseudo_ckpt.py --ckpt_dir ckpt/toy/
# Pseudo weights have been saved to path: ckpt/toy/consolidated.00.pth
```

Forward torch logits:

```shell
 torchrun --nproc_per_node 1 scripts/forward_torch.py  --ckpt_dir ckpt/toy/ 
```

Convert torch ckpt to paddle:

```shell
python scripts/conver_ckpt.py ckpt/toy/consolidated.00.pth
# If the weights are too large, it is recommended to use the following command, which will split the torch weights into a series of numpy files and finally merge them with paddle
python scripts/conver_ckpt_offline.py ckpt/toy/consolidated.00.pth ckpt/toy/np_ckpt
```

Forward paddle logits:

```shell
python -m paddle.distributed.launch  scripts/forward_paddle.py  --ckpt_dir ckpt/toy/ 
```

Check difference:

```shell
python scripts/check_diff.py forward_torch.npy forward_paddle.npy 
```

