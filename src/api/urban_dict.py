import requests
import csv


def get_word_definition(word):
    """
    Querieies the urban dictionary API for a given word, returning its definition.

    code referenced: https://www.geeksforgeeks.org/how-to-make-api-calls-using-python/

    Args:
        word (str): the word being searched for

    Returns:
        str: the words definition
    """
    # API endpoint URL
    url = f"https://unofficialurbandictionaryapi.com/api/search?term={word}&strict=true&matchCase=false&limit=1&page=1&multiPage=false&"

    try:
        # Make a GET request to the API endpoint using requests.get()
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            # remove un-needed data, returning only the definition
            return data["data"][0]["meaning"]
        else:
            print("Error:", response.status_code)
            return None

    except requests.exceptions.RequestException as e:
        # Handle any network-related errors or exceptions
        print("test")
        print("Error:", e)
        return None


def get_slang_words(phrase):
    """
    Finds all of the slang that the given phrase contains and returns them.

    Args:
        phrase (str): phrase to be used to check for any slang

    Returns:
        list: a list of all slang words/phrases in the given phrase
    """
    # initialize an empty set for matched slang
    matched_slang = set()

    # take the phrase and determine which words are in the slang corpus
    with open("../data/slang.csv", mode="r") as file:
        # create a CSV reader object
        csv_reader = csv.reader(file)

        # skip the header row
        next(csv_reader, None)

        # loop through each row in the CSV
        for row in csv_reader:
            slang_word = row[0]

            # if the slang word is in the phrase, add it to the matched slang set
            if slang_word in phrase:
                matched_slang.add(slang_word)

    # return the matched slang words as a list
    return list(matched_slang)


def slang_words_def(slang_contained):
    # initialize empty dictionary to hold words and their definitions
    word_defs = dict()

    # iterate through the slang_contained list and look up the definiions of each word
    for word in slang_contained:
        definition = get_word_definition(word)
        word_defs[word] = definition

    return word_defs
