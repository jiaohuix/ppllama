# ppllama
The paddle implementation of meta's [LLaMA](https://github.com/facebookresearch/llama).

## Setup

```
git clone https://github.com/MiuGod0126/ppllama.git
cd ppllama && pip install -r requirements.txt
pip install -e ./
```



## Inference

```shell
python -m paddle.distributed.launch  scripts/example.py --mp 1 --ckpt_dir ckpt/7B/ --tokenizer_path  ckpt/tokenizer.model
```

