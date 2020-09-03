import re

def normalize(text:str, with_space=False):
    text = text.lower().strip()

    if with_space:
        text = text.replace(' ','')

    text = re.sub('[-=.#/?:$}]', '', text)

    return text
