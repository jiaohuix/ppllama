import ipywidgets as widgets
from IPython.display import display, HTML
from translate import Translator
from ppllama import LLaMA
import requests
import random
import json
from hashlib import md5
# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


class BaiduTranslator:
    def __init__(self, appid, appkey, from_lang="zh", to_lang="en"):
        self.appid = appid
        self.appkey = appkey
        self.from_lang = from_lang
        self.to_lang = to_lang

        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        self.url = endpoint + path

    def translate(self, text):
        salt = random.randint(32768, 65536)
        sign = make_md5(self.appid + text + str(salt) + self.appkey)
        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': text, 'from': self.from_lang, 'to': self.to_lang, 'salt': salt, 'sign': sign}
        r = requests.post(self.url, params=payload, headers=headers)
        result = r.json()['trans_result'][0]['dst']

        return result

class SimpleChatbot:
    '''
        Usage:
        from ppllama import load_model,setup_model_parallel
        local_rank, world_size = setup_model_parallel()
        model, generator = load_model(ckpt_dir="ckpt/7B/", tokenizer_path="ckpt/tokenizer.model", local_rank=0, world_size=1)

        from examples.chatbot.ui import SimpleChatbot
        bot =  SimpleChatbot()
        bot.set_generator(generator)
        bot.show()

        # since translate lib is not stable, you can use baidu fanyi api
        # http://api.fanyi.baidu.com/product/11
        APPID= YOUR_APPID
        APPKEY= YOUR_APPKEY
        bot.switch_baidu_fanyi(appid=APPID, appkey=APPKEY)
        bot.show()
    '''
    def __init__(self,
                 usrname="User",
                 botname="Chatbot",
                 lang="en",
                 max_gen_len=128,
                 temperature=0.8,
                 top_p=0.95,
                 prompt=""):
        self.usrname = usrname
        self.botname = botname
        self.lang = lang
        # translator
        self.translator_fwd = Translator(from_lang=lang, to_lang="en") if lang != "en" else None
        self.translator_bwd = Translator(from_lang="en", to_lang=lang) if lang != "en" else None

        # generation
        self.generator = None
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p

        self.prompt = prompt

        # initialize ui
        self.init_ui()
        # bind button click event to function
        self.send_button.on_click(self.on_send_button_click)
        self.clear_button.on_click(self.on_clear_button_click)

    def on_send_button_click(self, button):
        with self.chat_history:
            # get user input
            user_message = self.user_input.value.strip()

            # display user message
            display(HTML(
                f'<p style="margin: 0 0 5px 0; font-weight: bold;">{self.usrname}:</p><p style="margin: 0 0 20px 0;">{user_message}</p>'))

            # TODO: Implement chatbot logic to generate reply
            chatbot_message = self.chat(self.preprocess(user_message))
            chatbot_message = self.postprocess(chatbot_message)

            # display chatbot message
            display(HTML(
                f'<p style="margin: 0 0 5px 0; font-weight: bold;">{self.botname}:</p><p style="margin: 0 0 20px 0;">{chatbot_message}</p>'))

            # clear user input
            self.user_input.value = ''

    def on_clear_button_click(self,button):
        self.chat_history.clear_output()

    def init_ui(self):
        self.layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            border='solid 1px',
            width='60%',
            height='500px',
            margin='10px'
        )
        # create user input box
        self.user_input = widgets.Text(
            placeholder='Type your message here',
            description=f"{self.usrname}:",
            layout=widgets.Layout(width='100%')
        )

        # create chat history box
        self.chat_history = widgets.Output(
            layout=widgets.Layout(flex="1", overflow_y='scroll')
        )

        # create send button
        self.send_button = widgets.Button(description='Send', layout=widgets.Layout(width='100px'))
        self.clear_button = widgets.Button(description='Clear', layout=widgets.Layout(width='100px'))

        # create chat UI
        self.chat_ui = widgets.VBox([self.chat_history, widgets.HBox([self.user_input, self.send_button,self.clear_button])], layout=self.layout)

    def show(self):
        # display chat UI
        display(self.chat_ui)


    def set_generator(self, generator):
        assert isinstance(generator,LLaMA), "generator must be instance of LLaMA"
        self.generator = generator
        print("Load generator over~")

    def chat(self, user_input):
        user_input = self.prompt + user_input
        bot_response = self.generator.generate([user_input],
                                          max_gen_len=self.max_gen_len,
                                          temperature=self.temperature,
                                          top_p=self.top_p)
        bot_response = bot_response[0][len(user_input):]
        return bot_response

    def preprocess(self, src_text):
        if self.translator_fwd is not None:
            src_text = self.translator_fwd.translate(src_text)
        return src_text

    def postprocess(self, tgt_text):
        if self.translator_bwd is not None:
            tgt_text = self.translator_bwd.translate(tgt_text)
        return tgt_text 

    def clear(self):
        self.chat_history.clear_output()

    def switch_baidu_fanyi(self, appid, appkey):
        self.translator_fwd = BaiduTranslator(appid,appkey, from_lang=self.lang, to_lang="en") if self.lang != "en" else None
        self.translator_bwd = BaiduTranslator(appid,appkey, from_lang="en", to_lang=self.lang) if self.lang != "en" else None
        print("Use baidu translator.")

