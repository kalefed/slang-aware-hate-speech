import os
import requests
import csv
import re


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
    # Tokenize the phrase into lowercase words
    # Tokenize the phrase into lowercase words
    words_in_phrase = set(
        re.findall(r"\b\w+\b", phrase.lower())
    )  # Tokenize and normalize to lowercase

    # Initialize an empty set for matched slang
    matched_slang = set()

    # Get the absolute path to the CSV file by navigating from the script's location
    script_dir = os.path.dirname(__file__)  # Directory where urban_dict.py is located
    csv_file_path = os.path.join(
        script_dir, "../../data/slang.csv"
    )  # Go up two directories to 'data'

    # Initialize a list to hold all slang terms (both multi-word and single-word)
    slang_terms = []

    # Read slang terms from CSV file
    with open(csv_file_path, mode="r") as file:
        csv_reader = csv.reader(file)

        # Skip the header row
        next(csv_reader, None)

        # Loop through each row in the CSV to populate the slang_terms list
        for row in csv_reader:
            slang_term = row[0].strip().lower()  # Normalize slang to lowercase
            slang_terms.append(slang_term)

    # Sort slang terms by length (longer phrases first)
    slang_terms.sort(key=lambda term: len(term.split()), reverse=True)

    # Check each slang term (multi-word first, then single-word)
    for slang_term in slang_terms:
        # If the phrase is in the words_in_phrase set, add it to matched_slang
        if slang_term in phrase.lower():
            matched_slang.add(slang_term)

    # Return the matched slang words as a list
    return list(matched_slang)


def slang_words_def(slang_contained):
    # initialize empty dictionary to hold words and their definitions
    word_defs = dict()

    # iterate through the slang_contained list and look up the definiions of each word
    for word in slang_contained:
        definition = get_word_definition(word)
        word_defs[word] = definition

    return word_defs
