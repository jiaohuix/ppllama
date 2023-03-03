# ppllama
This repo is the paddlepaddle implementation of meta's  [LLaMA](https://github.com/facebookresearch/llama) .



## Setup

```shell
git clone https://github.com/MiuGod0126/ppllama.git
cd ppllama && pip install -r requirements.txt
pip install -e ./
```

In order to download the checkpoints , fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5) (tokenizer already in [ckpt](./ckpt))

```shell
# download ckpt
bash scripts/download.sh <MODEL_SIZE>(7B/13B/30B/65B) <TARGET_FOLDER> <PRESIGNED_URL> 
```

The following is the checkpoints directory:

```
ckpt
├── 13B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   ├── consolidated.01.pth
│   └── params.json
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   ├── model0.pdparams
│   └── params.json
├── tokenizer_checklist.chk
└── tokenizer.model

```



## Quickstart

[Aistudio  ppllama ](https://aistudio.baidu.com/aistudio/projectdetail/5619235?channelType=0&channel=0)



## Alignment

This repository contains scripts for converting checkpoints from torch to paddle. I use a 2-layer model for inference to ensure that ppllama and llama are aligned, see:  [align](./examples/align/README.md)



## Inference

cpu: (50G RAM, 20min)

```shell
python -m paddle.distributed.launch  scripts/example_cpu.py --promt "The capital of Germany is the city of" --mp 1 --ckpt_dir ckpt/7B/ --tokenizer_path  ckpt/tokenizer.model
```

![ppllama cpu](https://i.328888.xyz/2023/03/04/FQejA.png)

gpu: (>40G )

```shell
python -m paddle.distributed.launch  scripts/example.py --mp 1 --ckpt_dir ckpt/7B/ --tokenizer_path  ckpt/tokenizer.model
```



If you like the project, please show your support by [leaving a star ⭐](https://github.com/MiuGod0126/ppllama/stargazers).





## License

See the [LICENSE](https://github.com/nebuly-ai/nebullvm/blob/main/apps/accelerate/chatllama/LICENSE) file.