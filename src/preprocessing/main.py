from api.urban_dict import get_slang_words, slang_words_def
import csv

word_list = get_slang_words("ok boomer")
word_def_dict = slang_words_def(word_list)

type_key = {
    "not_cyberbullying": "0",
    "religion": "1",
    "age": "2",
    "ethnicity": "3",
    "gender": "4",
    "other_cyberbullying": "5",
}


def reformat_data():
    """Reformats the data, replacing the type with an integer

    Returns:
        list: list of dictionaries, each dictiionary with the tweet label and text
    """
    # create an empty set to hold the data
    data = []

    with open("../data/cyberbullying_tweets.csv", mode="r") as csv_file:
        # create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # skip the header row
        next(csv_reader, None)

        # loop through each row in the CSV
        for row in csv_reader:
            entry = dict()
            tweet_text = row[0]
            tweet_type = row[1]
            entry["label"] = type_key[tweet_type]
            entry["text"] = tweet_text
            data.append(entry)

    return data


# we need to encode the labels
# split the data in train and validation sets
test = reformat_data()
print(test)
