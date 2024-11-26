from api.urban_dict import get_word_definition, slang_words_def, get_slang_words
import re
import demoji
import pandas as pd


class Preprocessing:
    def __init__(self):
        self.sentiment_mapping = {
            "not_cyberbullying": 0,
            "religion": 1,
            "age": 2,
            "ethnicity": 3,
            "gender": 4,
            "other_cyberbullying": 5,
        }

    @staticmethod
    def remove_urls(text):
        """Remove  URLs from the text.
        Args:
            text (string): text of the tweet

        Returns:
            string: text of the tweet
        """

        return re.sub(r"http\S+", "", text)

    @staticmethod
    def remove_mentions(text):
        """Remove mentions (e.g., @username) from the text.
        Args:
            text (string): text of the tweet

        Returns:
            string: text of the tweet
        """

        return re.sub(r"@\w+", "", text)

    @staticmethod
    def to_lowercase(text):
        """Converts text to lowercase.
        Args:
            text (string): text of the tweet

        Returns:
            string: text of the tweet
        """
        return text.lower()

    @staticmethod
    def convert_emojis(text):
        """Replace emojis with their textual descriptions

        Args:
            text (string): text of the tweet

        Returns:
            string: text of the tweet
        """
        emoji_map = demoji.findall(text)
        for emoji, desc in emoji_map.items():
            text = text.replace(emoji, f" {desc} ")
        return text

    def clean_text(self, text):
        """Apply all preprocessing steps to the text.

        Args:
            text (string): text of the tweet

        Returns:
            string: preprocessed tweet
        """

        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.to_lowercase(text)
        text = self.convert_emojis(text)
        return text

    def code_sentiment(self, df):
        """Replace sentiment labels in the DataFrame with numeric codes.
        Args:
            df (pandas dataframe): data

        Returns:
            pandas dataframe: updated DataFrame with coded sentiment
        """
        df["sentiment"] = df["sentiment"].replace(self.sentiment_mapping)
        return df
