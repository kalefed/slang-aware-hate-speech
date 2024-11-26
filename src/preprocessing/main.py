from api.urban_dict import get_slang_words, slang_words_def
import csv
import re
import demoji

type_key = {
    "not_cyberbullying": "0",
    "religion": "1",
    "age": "2",
    "ethnicity": "3",
    "gender": "4",
    "other_cyberbullying": "5",
}

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = text.lower() #lowercase
    text = remove_emoji(text) # Convert all emojis to words
    return text

def remove_emoji(text):
    emoji_map = demoji.findall(text)
    for emoji, desc in emoji_map.items():
        text = text.replace(emoji, f" {desc} ")
    return text

def reformat_data():
    """Reformats the data, replacing the type with an integer

    Returns:
        list: list of dictionaries, each dictiionary with the tweet label and text
    """
    # create an empty set to hold the data
    data = []

    with open("../data/testtweets.csv", mode="r") as csv_file:
        # create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # skip the header row
        next(csv_reader, None)
        # loop through each row in the CSV
        for row in csv_reader:
            entry = dict()
            tweet_text = row[0]
            tweet_type = row[1]

            # Clean the tweet text
            clean_tweet_text = clean_text(tweet_text)
            
            # Identify slang words in the tweet
            slang_words = get_slang_words(clean_tweet_text)
            
            # Get definitions for the slang words
            if len(slang_words) > 0:
                slang_definitions = slang_words_def(slang_words)
                # Append definitions to the text
                definitions_text = " ".join([f"{word}: {definition}" for word, definition in slang_definitions.items() if definition])
                clean_tweet_text = f"{clean_tweet_text} [{definitions_text}]"


            entry["label"] = type_key[tweet_type]
            entry["text"] = clean_tweet_text
            data.append(entry)

    return data


# we need to encode the labels
# split the data in train and validation sets
test = reformat_data()
print(test)
