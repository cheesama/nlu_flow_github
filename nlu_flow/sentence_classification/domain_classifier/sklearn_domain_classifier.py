from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from tqdm import tqdm
from pprint import pprint

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file))))))

from utils import meta_db_client

def load_dataset(
    table_list=[
        "intent-rules",
        "nlu-chitchat-utterances",
        "nlu-faq-questions",
        "nlu-intent-entity-utterances",
        "nlu-slang-utterances",
    ]
):
    pass
