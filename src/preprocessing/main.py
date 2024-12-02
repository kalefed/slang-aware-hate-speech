from api.urban_dict import get_word_definition, slang_words_def, get_slang_words
import re
import demoji
import pandas as pd
import torch
from transformers import DistilBertTokenizer

pd.set_option("future.no_silent_downcasting", True)  # fix .replace error


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

    @staticmethod
    def add_slang_def(text):
        slang_contained = get_slang_words(text)
        word_defs = slang_words_def(slang_contained)

        # Loop through the slang words and add their definitions to the text
        for slang, definition in word_defs.items():
            # Create a formatted string to include the definition
            formatted_slang = f"{slang} (meaning: {definition})"

            # Replace the slang word in the text with the formatted string
            # Using re.sub to replace the exact slang word (case-insensitive)
            text = re.sub(
                rf"\b{re.escape(slang)}\b", formatted_slang, text, flags=re.IGNORECASE
            )

        return text

    @staticmethod
    def clean_tweet(self, text):
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
        text = self.add_slang_def(text)
        return text

    def clean_tweets(self, df):
        """Cleans the tweet text and adds a new row to the dataframe with the cleaned text.

        Args:
            df (DataFrame): data to be used
        """
        df["text_clean"] = [self.clean_tweet(self, tweet) for tweet in df["tweet_text"]]
        return df

    def code_sentiment(self, df):
        """Replace sentiment labels in the DataFrame with numeric codes.
        Args:
            df (pandas dataframe): data

        Returns:
            pandas dataframe: updated DataFrame with coded sentiment
        """
        df["cyberbullying_type"] = (
            df["cyberbullying_type"].replace(self.sentiment_mapping).astype(int)
        )
        return df

    def tokenizer(self, data, max_length=512):
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", do_lower_case=True
        )
        input_ids = []

        attention_masks = []
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]` special tokens
                max_length=max_length,  # Choose max length to truncate/pad
                padding="max_length",
                return_attention_mask=True,  # Return attention mask
                truncation=True,
            )
            input_ids.append(encoded_sent.get("input_ids"))
            attention_masks.append(encoded_sent.get("attention_mask"))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def convert_to_tensors(self, y_train_os, y_valid, y_test):
        train_labels = torch.from_numpy(y_train_os)
        val_labels = torch.from_numpy(y_valid)
        test_labels = torch.from_numpy(y_test)

        return train_labels, val_labels, test_labels