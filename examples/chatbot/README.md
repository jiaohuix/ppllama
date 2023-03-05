# SimpleChatbot

Building a simple chatbot in jupyter using ipywidgets

```shell
pip install ipywidgets translate
```



```shell
from ppllama.examples.ui import SimpleChatbot
bot =  SimpleChatbot(
				 usrname="User",
                 botname="Chatbot",
                 lang="en",
                 max_gen_len=64,
                 temperature=0.8,
                 top_p=0.95,
                 prompt=None)
bot.load_model(ckpt_dir="ckpt/7B/", tokenizer_path="ckpt/tokenizer.model")
bot.display()
```

