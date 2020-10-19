import re

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile("<.*?>")

    return re.sub(clean, "", text)


def normalize(text: str, with_space=False):
    text = text.lower().strip()
    text = remove_html_tags(text)

    if with_space:
        text = text.replace(" ", "")

    text = re.sub("[-=.#/?:$}]", "", text)

    return text
