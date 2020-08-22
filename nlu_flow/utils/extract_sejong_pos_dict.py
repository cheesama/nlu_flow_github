from sejong_corpus_cleaner import get_data_paths
from sejong_corpus_cleaner import write_sentences
from sejong_corpus_cleaner import Sentences
from sejong_corpus_cleaner import to_simple_tag
from sejong_corpus_cleaner import make_lr_corpus
from sejong_corpus_cleaner import load_a_sejong_file

from soynlp.postagger.tagset import tagset

from tqdm import tqdm

import dill

paths = get_data_paths(corpus_types='written') # 문어체 말뭉치만 가져올 경우
sents = Sentences(paths, verbose=True)
write_sentences(sents, 'sejong_corpus.txt')

sents = load_a_sentences_file('sejong_corpus.txt')

for tag in 'NNB NNG NNP XR XSN NR EC EF JC JKB SH NNNG'.split():
    print('{} -> {}'.format(tag, to_simple_tag(tag)))

make_lr_corpus(sents, filepath='lr_corpus_type1.txt')

pos_dict = {}
for key in tagset.keys():
    pos_dict[key] = set()

with open('lr_corpus_type1.txt', 'r') as pos_data:
    lines = pos_data.readlines()
    for line in tqdm(line, desc='storing POS data...'):
        if len(line.strip()) < 1: continue
        if len(line.split('\t')) < 2: continue

        for each_pos in line.split('\t')[1].strip().split('+'):
            each_pos = each_pos.strip()
            if len(each_pos.split('/')) < 2: break

            value = each_pos.split('/')[0]
            pos = each_pos.split('/')[1]

            if pos in tagset.keys():
                pos_dict[pos].add(value)

with open('sejong_pos_dict.dill', 'wb') as dict_file:
    dill.dump(pos_dict, dict_file)

