import requests
import random
import json
from hashlib import md5
import time
import threading
import ipywidgets as widgets
from IPython.display import display, HTML
from translate import Translator
from ppllama import LLaMA
import requests
import random
import json
from hashlib import md5
from collections import namedtuple


# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


def get_time():
    return time.strftime("%H:%M", time.localtime()).lstrip("0")


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
        payload = {'appid': self.appid, 'q': text, 'from': self.from_lang, 'to': self.to_lang, 'salt': salt,
                   'sign': sign}
        r = requests.post(self.url, params=payload, headers=headers)
        result = r.json()['trans_result'][0]['dst']

        return result


bubble_css = '''
    /* QuickReset */ * {margin: 0; box-sizing: border-box;}

    .chat {
    --rad: 20px;
    --rad-sm: 3px;
    font: 16px/1.5 sans-serif;
    display: flex;
    flex-direction: column;
    padding: 20px;
    margin: auto;
    }

    .msg {
    position: relative;
    max-width: 75%;
    padding: 7px 15px;
    margin-bottom: 2px;
    }

    .msg.sent {
    border-radius: var(--rad) var(--rad-sm) var(--rad-sm) var(--rad);
    background: #42a5f5;
    color: #fff;
    /* moves it to the right */
    margin-left: auto;
    }

    .msg.rcvd {
    border-radius: var(--rad-sm) var(--rad) var(--rad) var(--rad-sm);
    background: #f1f1f1;
    color: #555;
    /* moves it to the left */
    margin-right: auto;
    }

    /* Improve radius for messages group */

    .msg.sent:first-child,
    .msg.rcvd+.msg.sent {
    border-top-right-radius: var(--rad);
    }

    .msg.rcvd:first-child,
    .msg.sent+.msg.rcvd {
    border-top-left-radius: var(--rad);
    }


    /* time */

    .msg::before {
    content: attr(data-time);
    font-size: 0.8rem;
    position: absolute;
    bottom: 100%;
    color: #888;
    white-space: nowrap;
    /* Hidden by default */
    display: none;
    }

    .msg.sent::before {
    right: 15px;
    }

    .msg.rcvd::before {
    left: 15px;
    }


    /* Show time only for first message in group */

    .msg:first-child::before,
    .msg.sent+.msg.rcvd::before,
    .msg.rcvd+.msg.sent::before {
    /* Show only for first message in group */
    display: block;
    }
'''

class SimpleChatbot:
    '''
        Usage:
            # step1: load generator
            from ppllama import load_model,setup_model_parallel
            from examples.chatbot.ui import SimpleChatbot
            local_rank, world_size = setup_model_parallel()
            model, generator = load_model(ckpt_dir="ckpt/7B/", tokenizer_path="ckpt/tokenizer.model", local_rank=0, world_size=1)

            # setp2: build chatbot ui
            bot =  SimpleChatbot(usrname="User",
                                botname="Chatbot",
                                lang="zh",
                                max_gen_len=128,
                                temperature=0.8,
                                top_p=0.95,
                                prompt=None)
            bot.set_generator(generator)
            bot.show()
            # bot.clear()  # clear chat history

            # step3: use baidu translator (optional)  https://fanyi-api.baidu.com/product/11
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

        # record history
        self.CHAT_PAIR = namedtuple("Pair",["send","recv"])
        self.chatbot_state = []

    def on_send_button_click(self, button):
        with self.chat_history:
            # get user input
            user_message = self.user_input.value.strip()

            # display user message
            # author：https://juejin.cn/post/7021426688634388487
            # max-width: 500px;  chat框占屏幕宽度

            # user_html = f'''
            # <p style="margin: 0 5px 5px 0; font-weight: bold;text-align:right;">{self.usrname}</p>
            # <p class="chat-right_triangle" style="margin: 0 5px 20px 0;text-align:right;">{user_message}</p>
            # '''
            # https://stackoverflow.com/questions/71154905/css-for-chat-room-speech-bubble-position
            user_html = f'''
                        <div class="chat">
                        <div data-time="{self.usrname} {get_time()}" class="msg sent"> {user_message} </div>
                        </div>
            '''
            user_html_with_css = '<style>{}</style>{}'.format(bubble_css, user_html)

            display(HTML(user_html_with_css))

            # TODO: Implement chatbot logic to generate reply
            self.chat(user_message)
            # thread = threading.Thread(target=self.chat, args=(user_message,))
            # thread.start()
            cur_state = self.chatbot_state[-1]
            chatbot_message = cur_state.recv

            # display chatbot message
            bot_html = f'''
                        <div class="chat">
                        <div data-time="{self.botname} {get_time()}" class="msg rcvd"> {chatbot_message} </div>
                        </div>
            '''
            bot_html_with_css = '<style>{}</style>{}'.format(bubble_css, bot_html)
            display(HTML(bot_html_with_css))
            # thread.join()

            # display(HTML( f'<p style="margin: 0 0 20px 0;">{chatbot_message}</p><p style="margin: 0 0 5px 0; font-weight: bold;">{self.botname}:</p>'))

            # clear user input
            self.user_input.value = ''

    def on_clear_button_click(self, button):
        self.chat_history.clear_output()

    def on_len_change(self, change):
        self.max_gen_len = change["new"]

    def on_temperature_change(self, change):
        print(change["new"])
        self.temperature = change["new"]

    def on_top_p_change(self, change):
        self.top_p = change["new"]

    def init_ui(self):
        self.layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            border='solid 1px',
            width='75%',
            height='600px',
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

        # create parameters slider
        self.len_slider = widgets.IntSlider(value=128, min=0, max=1024, step=32, description="Max_len:", layout=widgets.Layout(width='90%'))
        self.temperature_slider = widgets.FloatSlider(value=0.8, min=0, max=10.0, step=0.1, description="Temperature:", layout=widgets.Layout(width='90%'))
        self.top_p_slider = widgets.FloatSlider(value=0.95, min=0, max=1.0, step=0.1, description="Top_p:", layout=widgets.Layout(width='90%'))
        self.len_slider.observe(self.on_len_change, names="value")
        self.temperature_slider.observe(self.on_temperature_change, names="value")
        self.top_p_slider.observe(self.on_top_p_change, names="value")
        # How to increase slider length in ipywidgets: https://stackoverflow.com/questions/63897367/how-to-increase-slider-length-in-ipywidgets
        # create chat UI
        self.chat_ui = widgets.VBox([
            self.chat_history,
            widgets.HBox([self.user_input, self.send_button, self.clear_button]),
            widgets.VBox([self.len_slider, self.temperature_slider, self.top_p_slider])], layout=self.layout)

    def show(self):
        # display chat UI
        display(self.chat_ui)

    def set_generator(self, generator):
        assert isinstance(generator, LLaMA), "Generator must be instance of LLaMA"
        self.generator = generator
        print("Load generator over~")

    def chat(self, user_input):
        if self.generator is  None:
            state = self.CHAT_PAIR(send=user_input, recv="Generator should not be none.")
            self.chatbot_state.append(state)
            return
        user_input = self.preprocess(user_input)
        user_input = self.prompt + user_input
        bot_response = self.generator.generate([user_input],
                                               max_gen_len=self.max_gen_len,
                                               temperature=self.temperature,
                                               top_p=self.top_p)
        bot_response = bot_response[0][len(user_input):]
        bot_responses = [self.postprocess(bot_response) for resp in bot_response.split('\n')]
        bot_response = "\n".join(bot_responses)
        state = self.CHAT_PAIR(send=user_input, recv=bot_response)
        self.chatbot_state.append(state)

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

    def switch_baidu_fanyi(self, appid=None, appkey=None):
        assert appid is not None and appkey is not None, "appid or appkey should not be None, " \
                                                         "please apply on the Baidu Translation Open Platform:" \
                                                         "https://fanyi-api.baidu.com/product/11 "
        self.translator_fwd = BaiduTranslator(appid, appkey, from_lang=self.lang,
                                              to_lang="en") if self.lang != "en" else None
        self.translator_bwd = BaiduTranslator(appid, appkey, from_lang="en",
                                              to_lang=self.lang) if self.lang != "en" else None
        print("Use baidu translator.")

