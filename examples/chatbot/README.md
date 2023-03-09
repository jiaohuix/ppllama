# SimpleChatbot

Building a simple chatbot in jupyter using ipywidgets

```shell
pip install ipywidgets translate
```

1. load generator:

```python
from ppllama import load_model,setup_model_parallel
local_rank, world_size = setup_model_parallel()
model, generator = load_model(ckpt_dir="ckpt/7B/", tokenizer_path="ckpt/tokenizer.model", local_rank=0, world_size=1)

```

2.  load chatbot

```python
from examples.chatbot.ui import SimpleChatbot
bot =  SimpleChatbot(usrname="User",
                    botname="Chatbot",
                    lang="zh",
                    max_gen_len=64,
                    temperature=0.8,
                    top_p=0.95,
                    prompt=None)
bot.set_generator(generator)
bot.show()
# bot.clear()  # clear chat history
```

3.  use baidu translator (optional):

```python
APPID= YOUR_APPID
APPKEY= YOUR_APPKEY
bot.switch_baidu_fanyi(appid=APPID, appkey=APPKEY)
bot.show()
```

![](https://ai-studio-static-online.cdn.bcebos.com/27af144d45de4b7bb3cb4d80ae82cec84a0be7e9af8a4251908d35f8c40d6697)