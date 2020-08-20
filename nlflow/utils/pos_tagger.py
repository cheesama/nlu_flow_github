from soynlp.postagger import Dictionary
from soynlp.postagger import LRTemplateMatcher
from soynlp.postagger import LREvaluator
from soynlp.postagger import SimpleTagger
from soynlp.postagger import UnknowLRPostprocessor
import soynlp

from tqdm import tqdm

import os, sys

class PosTagger:
    def __init__(self):
        #set initial pos_dict from installed package location
        self.pos_dict = dict()
        pos_dict_default_location = soynlp.__file__[:soynlp.__file__.rfind(os.sep)]
        pos_dict_candidate_folders = ['pos', 'postagger', 'noun', 'lemmatizer']

        for folder in pos_dict_candidate_folders:
            pos_dict_location  = pos_dict_default_location + os.sep + folder

            for root,dirs, files in os.walk(pos_dict_location):
                for pos_file in files:
                    if 'dictionary' not in root + os.sep + pos_file or '.txt' not in pos_file: continue
                    #print (root + os.sep + pos_file)
                    pos_type = pos_file.replace('.txt','').capitalize()
                    if pos_type not in self.pos_dict.keys():
                        self.pos_dict[pos_type] = set()
                    with open(root + os.sep + pos_file, 'r') as pos_values:
                        values = pos_values.readlines()
                        for value in tqdm(values, desc=f'setting default POS dict: {pos_file}'):
                            self.pos_dict[pos_type].add(value.strip())

        print (f'POS type : {self.pos_dict.keys()}')

        self.dictionary = Dictionary(self.pos_dict)
        self.generator = LRTemplateMatcher(self.dictionary)    
        self.evaluator = LREvaluator()
        self.postprocessor = UnknowLRPostprocessor()
        self.tagger = SimpleTagger(self.generator, self.evaluator, self.postprocessor)

    def tag(self, sentence):
        return self.tagger.tag(sentence)

    def extract_noun(self, sentences:list):
        pass

    def add_words(self, tag:str, words:list):
        pass

    def get_vocab(self):
        pass

