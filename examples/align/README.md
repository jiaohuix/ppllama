# align

```shell
# in ppllama folder
pip install reprod_log tqdm
git clone https://github.com/facebookresearch/llama.git
cd llama && pip install -r requirements.txt
pip install -e ./
```

Generate random weights for two layers:

```shell
torchrun --nproc_per_node 1 scripts/get_pseudo_ckpt.py --ckpt_dir ckpt/toy/ --tokenizer_path ckpt/tokenizer.model 
# Pseudo weights have been saved to path: ckpt/toy/consolidated.00.pth
```

Forward torch logits:

```shell
torchrun --nproc_per_node 1 scripts/forward_torch.py  --ckpt_dir ckpt/toy/ --tokenizer_path ckpt/tokenizer.model 
```

Convert torch ckpt to paddle:

```shell
# method1:
python scripts/conver_ckpt.py ckpt/toy/consolidated.00.pth

# method2: ★
# If the weights are too large, it is recommended to use the following command, which will split the torch weights into a series of numpy files and partially load state.
python scripts/conver_ckpt_offline.py ckpt/toy/consolidated.00.pth ckpt/toy/consolidated.00.pdparams  n
```

Forward paddle logits:

```shell
python -m paddle.distributed.launch  scripts/forward_paddle.py  --ckpt_dir ckpt/toy/ --tokenizer_path ckpt/tokenizer.model 
```

Check difference:

```shell
python scripts/check_diff.py forward_torch.npy forward_paddle.npy 
```

```shell
[2023-03-05 10:50:51,830] [    INFO] utils.py:118 - input_tokens: 
[2023/03/05 10:50:51] root INFO: input_tokens: 
[2023-03-05 10:50:51,831] [    INFO] utils.py:123 -     mean diff: check passed: True, value: 0.0
[2023/03/05 10:50:51] root INFO:        mean diff: check passed: True, value: 0.0
[2023-03-05 10:50:51,831] [    INFO] utils.py:118 - logits: 
[2023/03/05 10:50:51] root INFO: logits: 
[2023-03-05 10:50:51,831] [    INFO] utils.py:123 -     mean diff: check passed: True, value: 3.0453648491857166e-07
[2023/03/05 10:50:51] root INFO:        mean diff: check passed: True, value: 3.0453648491857166e-07
[2023-03-05 10:50:51,831] [    INFO] ReprodDiffHelper.py:64 - diff check passed
[2023/03/05 10:50:51] root INFO: diff check passed
```

