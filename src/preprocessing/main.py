from api.urban_dict import get_word_definition, slang_words_def, get_slang_words
import csv
import re
import demoji

TYPE_KEY = {
    "not_cyberbullying": 0,
    "religion": 1,
    "age": 2,
    "ethnicity": 3,
    "gender": 4,
    "other_cyberbullying": 5,
}


def remove_urls(text):
    re.sub(r"http\S+", "", text)
    return text


def remove_mentions(text):
    re.sub(r"@\w+", "", text)
    return text


def to_lowercase(text):
    text.lower()
    return text


def convert_emojis(text):
    emoji_map = demoji.findall(text)

    # replace all emojis with their description
    for emoji, desc in emoji_map.items():
        text = text.replace(emoji, f" {desc} ")
    return text


def clean_text(text):
    text = remove_urls(text)
    text = remove_mentions(text)
    text = to_lowercase(text)
    text = convert_emojis(text)
    return text


def code_sentiment(df):
    df["sentiment"] = df["sentiment"].replace(TYPE_KEY)
