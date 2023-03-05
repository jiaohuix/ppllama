import ipywidgets as widgets
from IPython.display import display, HTML
from translate import Translator
from functools import partial
from ppllama import load_model, setup_model_parallel


class SimpleChatbot:
    '''
        Usage:
        from ppllama.examples.ui import SimpleChatbot
        bot =  SimpleChatbot()
        bot.load_model(ckpt_dir="ckpt/7B/", tokenizer_path="ckpt/tokenizer.model")
        bot.display()
    '''
    def __init__(self,
                 usrname="User",
                 botname="Chatbot",
                 lang="en",
                 max_gen_len=64,
                 temperature=0.8,
                 top_p=0.95,
                 prompt=None):
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

        self.prompt = "You are asked to pretend to be a human being in response to the user's " \
                      "input, which is as follows:" if prompt is not None else prompt

        # initialize ui
        self.init_ui()

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

        # bind button click event to function
        self.send_button.on_click(on_send_button_click)

        # create chat UI
        self.chat_ui = widgets.VBox([chat_history, widgets.HBox([user_input, send_button])], layout=layout)

    def display(self):
        # display chat UI
        display(self.chat_ui)

    def on_send_button_click(self, button):
        with chat_history:
            # get user input
            user_message = user_input.value.strip()

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
            user_input.value = ''

    def load_model(self, ckpt_dir="ckpt/7B/", tokenizer_path="ckpt/tokenizer.model"):
        local_rank, world_size = setup_model_parallel()
        model, generator = load_model(ckpt_dir=ckpt_dir,
                                      tokenizer_path=tokenizer_path,
                                      local_rank=0,
                                      world_size=1)
        self.generator = generator

    def chat(self, user_input):
        user_input = self.prompt + user_input
        bot_response = generator.generate([user_input],
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
            tgt_text = self.translator_fwd.translate(tgt_text)
        return tgt_text

