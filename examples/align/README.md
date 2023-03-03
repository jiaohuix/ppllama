# align

```shell
pip install reprod_log tqdm
```

Generate random weights for two layers:

```shell
torchrun --nproc_per_node 1 scripts/get_pseudo_ckpt.py --ckpt_dir ckpt/toy/
# Pseudo weights have been saved to path: ckpt/toy/consolidated.00.pth
```

Forward torch:

```shell
 torchrun --nproc_per_node 1 scripts/forward_torch.py  --ckpt_dir ckpt/toy/ 
```

Convert torch ckpt to paddle:

```shell
python scripts/conver_ckpt.py ckpt/toy/consolidated.00.pth
```

Forward paddle:

```shell
python -m paddle.distributed.launch  scripts/forward_paddle.py  --ckpt_dir ckpt/toy/ 
```

Check diff:

```shell
python scripts/check_diff.py forward_torch.npy forward_paddle.npy 
```

