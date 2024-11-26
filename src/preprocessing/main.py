from api.urban_dict import get_word_definition, slang_words_def, get_slang_words
import re
import demoji
import pandas as pd
import torch
from transformers import BertTokenizer


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
        return text  # TODO - should we add this text here as a new row or should that be done in another function??

    def add_clean_text(self):
        # TODO - implement this function! possibly adds clean_text row to data
        pass

    def code_sentiment(self, df):
        """Replace sentiment labels in the DataFrame with numeric codes.
        Args:
            df (pandas dataframe): data

        Returns:
            pandas dataframe: updated DataFrame with coded sentiment
        """
        df["cyberbullying_type"] = df["cyberbullying_type"].replace(
            self.sentiment_mapping
        )
        return df

    def tokenizer(self, data, max_length=128):
        # TODO - this code is from kaggle file so should look over and change if needed
        # TODO - Add docstring once done
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        input_ids = []

        attention_masks = []
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]` special tokens
                max_length=max_length,  # Choose max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                return_attention_mask=True,  # Return attention mask
            )
            input_ids.append(encoded_sent.get("input_ids"))
            attention_masks.append(encoded_sent.get("attention_mask"))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def convert_to_tensors(y_train_os, y_valid, y_test):
        # TODO - add docstring & determine if we need the 'y-valid' data
        train_labels = torch.from_numpy(y_train_os)
        val_labels = torch.from_numpy(y_valid)
        test_labels = torch.from_numpy(y_test)
