from pynori.korean_analyzer import KoreanAnalyzer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from itertools import chain

import re, json, requests

nori = KoreanAnalyzer(
    decompound_mode="DISCARD",  # DISCARD or MIXED or NONE
    infl_decompound_mode="DISCARD",  # DISCARD or MIXED or NONE
    discard_punctuation=True,
    output_unknown_unigrams=False,
    pos_filter=False,
    stop_tags=["JKS", "JKB", "VV", "EF"],
    synonym_filter=False,
    mode_synonym="NORM",
)  # NORM or EXTENSION

# print(nori.do_analysis("아빠가 방에 들어가신다."))
def tokenize(text):
    return nori.do_analysis(text)["termAtt"]


def tokenize_fn(text):
    return nori.do_analysis(text)

def convert_ner_data_format(text:str, entities:list, tokenize_fn=tokenize_fn):
    tokens = tokenize_fn(text)

    entity_types = []
    entity_values = []
    entity_positions = []

    entity_dataset = []

    for data in entities:
        entity_type = data['entity']
        entity_value = data['value']
        entity_position = (data['start'], data['end'])

        entity_types.append(entity_type)
        entity_values.append(entity_value)
        entity_positions.append(entity_position)


    for i, token_position in enumerate(tokens["offsetAtt"]):
        entity_data = [tokens["termAtt"][i], tokens["posTagAtt"][i], "O"]

        for i, entity_position in enumerate(entity_positions):
            if (
                entity_position[0] <= token_position[0]
                and token_position[1] <= entity_position[1]
            ):
                if entity_position[0] == token_position[0]:
                    entity_tag = "B-" + entity_types[i]
                else:
                    entity_tag = "I-" + entity_types[i]

                entity_data[2] = entity_tag
                break

        entity_dataset.append(tuple(entity_data))

    # print (entity_dataset)
    return entity_dataset


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        "bias",
        "word.lower=" + word.lower(),
        "word.isdigit=%s" % word.isdigit(),
        "postag=" + postag,
    ]

    if i > 1:
        word1 = sent[i - 2][0]
        postag1 = sent[i - 2][1]
        features.extend(
            [
                "-2:word.lower=" + word1.lower(),
                "-2:word.isdigit=%s" % word1.isdigit(),
                "-2:postag=" + postag1,
            ]
        )

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend(
            [
                "-1:word.lower=" + word1.lower(),
                "-1:word.isdigit=%s" % word1.isdigit(),
                "-1:postag=" + postag1,
            ]
        )
    else:
        features.append("BOS")

    if i < len(sent) - 2:
        word1 = sent[i + 2][0]
        postag1 = sent[i + 2][1]
        features.extend(
            [
                "+2:word.lower=" + word1.lower(),
                "+2:word.isdigit=%s" % word1.isdigit(),
                "+2:postag=" + postag1,
            ]
        )

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend(
            [
                "+1:word.lower=" + word1.lower(),
                "+1:word.isdigit=%s" % word1.isdigit(),
                "+1:postag=" + postag1,
            ]
        )
    else:
        features.append("EOS")

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def bio_classification_report(y_true, y_pred, output_dict=True):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {"O"}
    tagset = sorted(tagset, key=lambda tag: tag.split("-", 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
        output_dict=output_dict
    )

def slack_report(webhook_url, file_name='report.md', title='Intent & Entity Validation Report'):
    content = open(file_name).readlines()
    content = ''.join(content)

    payload = {
        "text": title,
        "attachments": [
            {
                "color": 'red',
                "fields": [
                    {
                        "title": "Check Below Result",
                        "value": content,
                        "short": False
                    }
                ]
            }
        ]
    }

    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
