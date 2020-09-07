from tqdm import tqdm

import os, sys
import re
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from preprocessor.text_preprocessor import normalize

class KorCharTokenizer:
    '''
    a ~ z : 97 ~ 122
    가 ~ 힣 : 44032 ~ 55203
    '''
    def __init__(self, max_len=128, padding=True):
        self.max_len = 128
        self.padding = padding

        self.char_dict = {}
        self.char_dict[0] = '[CLS]'
        self.char_dict[1] = '[SEP]'
        self.char_dict[2] = '[PAD]'
        self.char_dict[3] = '[UNK]'
        self.char_dict[4] = ' '

        # a ~ z mapping
        for i in range(97, 123):
            self.char_dict[len(self.char_dict)] = chr(i)

        # 가 ~ 힣 mapping
        for i in tqdm(range(44032, 55204), desc='building korean char_token dictionary...'):
            self.char_dict[len(self.char_dict)] = chr(i)

        self.char_token_dict = {}
        for k, v in self.char_dict.items():
            self.char_token_dict[v] = k

    def tokenize(self, text):
        text = normalize(text)
        tokens = []
        tokens.append(0) # append CLS token default as BOS

        for char in text:
            if 97 <= ord(char) <=122 or 44032 <= ord(char) <= 55203:
                tokens.append(self.char_token_dict[char])
            else:
                tokens.append(4) #unknown token

        tokens.append(1) # append SEP token default as EOS

        if len(tokens) < self.max_len:
            tokens +=  ([2] * (sel.max_len - len(tokens)))



