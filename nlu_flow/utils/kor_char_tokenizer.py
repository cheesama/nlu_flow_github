from tqdm import tqdm

import os, sys
import re

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from preprocessor.text_preprocessor import normalize

PAD_TOKEN_ID = 2


class KorCharTokenizer:
    """
    a ~ z : 97 ~ 122
    가 ~ 힣 : 44032 ~ 55203
    """

    def __init__(self, max_len=128, padding=True):
        self.max_len = 128
        self.padding = padding

        self.char_dict = {}
        self.char_dict[0] = "[CLS]"
        self.char_dict[1] = "[SEP]"
        self.char_dict[PAD_TOKEN_ID] = "[PAD]"
        self.char_dict[3] = "[UNK]"
        self.char_dict[4] = " "
        self.char_dict[5] = "!"
        self.char_dict[6] = "?"
        self.char_dict[7] = "^"

        # 0 ~ 9 mapping
        for i in range(48, 58):
            self.char_dict[len(self.char_dict)] = chr(i)

        # A ~ Z mapping
        for i in range(65, 91):
            self.char_dict[len(self.char_dict)] = chr(i)

        # a ~ z mapping
        for i in range(97, 123):
            self.char_dict[len(self.char_dict)] = chr(i)

        # 가 ~ 힣 mapping
        for i in tqdm(
            range(44032, 55204), desc="building korean char_token dictionary..."
        ):
            self.char_dict[len(self.char_dict)] = chr(i)

        self.char_token_dict = {}
        for k, v in self.char_dict.items():
            self.char_token_dict[v] = k

    def tokenize(self, text, padding=True, norm=False):
        if norm:
            text = normalize(text)
        tokens = [0]  # append CLS token default as BOS

        for char in text:
            char = str(char)[0]

            if (
                32 <= ord(char) < 34            # ' ' and '!'
                or 48 <= ord(char) < 58
                or 63 == ord(char)              # '?'
                or 65 <= ord(char) < 91
                or 97 <= ord(char) < 123
                or 44032 <= ord(char) < 55204
            ):
                tokens.append(self.char_token_dict[char])
            else:
                tokens.append(3)  # unknown token

        tokens.append(1)  # append SEP token default as EOS

        if padding and len(tokens) < self.max_len:
            tokens += [PAD_TOKEN_ID] * (self.max_len - len(tokens))

        # print (tokens[:self.max_len])

        return tokens[: self.max_len]

    def decode(self, tokens):
        result = ""
        for token in tokens:
            if token == 1:  # means EOS
                break

            if 0 <= token <= 3:
                continue

            result += self.char_dict[token]

        return result

    def get_vocab_size(self):
        return len(self.char_dict)

    def get_pad_token_id(self):
        return PAD_TOKEN_ID

    def get_seq_len(self):
        return self.max_len
