# from .chatgpt import ChatGPT
# from .alpaca import Alpaca
# from .longchat.longchat import Longchat
# from .flan_t5 import FlanT5
from .base_language_model import BaseLanguageModel
from .llama import Llama2, RoG, Llama3, RoG_Llama3

registed_language_models = {
    # 'gpt-4': ChatGPT,
    # 'gpt-3.5-turbo': ChatGPT,
    # 'alpaca': Alpaca,
    # 'longchat': Longchat,
    # 'flan-t5': FlanT5,
    'llama2': Llama2,
    'rog': RoG,
    "llama3": Llama3,
    "rog-llama3": RoG_Llama3
}


def get_registed_model(model_name) -> BaseLanguageModel:
    for key, value in registed_language_models.items():
        if key == model_name.lower():
            return value
    raise ValueError(f"No registered model found for name {model_name}")
