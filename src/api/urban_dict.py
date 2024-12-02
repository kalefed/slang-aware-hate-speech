import os
import requests
import re
import functools

SLANG_WORDS_SET = {
    "cool",
    "lit",
    "dope",
    "fire",
    "savage",
    "af",
    "cap",
    "cheugy",
    "cringe",
    "dead",
    "extra",
    "fit",
    "goat",
    "go off",
    "gucci",
    "hits different",
    "irl",
    "iykyk",
    "low-key",
    "hear me out",
    "omg",
    "ong",
    "preppy",
    "rizz",
    "salty",
    "sick",
    "sleep on",
    "snatched",
    "tbh",
    "tea",
    "vanilla",
    "thirsty",
    "yeet",
    "yassify",
    "delulu",
    "skibidi",
    "sigma",
    "drip",
    "bussin",
    "gyat",
    "mewing",
    "fanum tax",
    "bet",
    "sus",
    "basic",
    "it's giving",
    "bop",
    "big yikes",
    "glow-up",
    "ohio",
    "period",
    "periodt",
    "ick",
    "mid",
    "big w",
    "big l",
    "asf",
    "asl",
    "aura",
    "based",
    "bde",
    "bffr",
    "boujee",
    "brainrot",
    "caught in 4k",
    "clapback",
    "cook",
    "dab",
    "delusionship",
    "dogs",
    "era",
    "finna",
    "gagged",
    "ghost",
    "glaze",
    "ipad kid",
    "i oop",
    "karen",
    "main character",
    "mogging",
    "npc",
    "ok boomer",
    "out of pocket",
    "waste of space",
}


@functools.cache
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
    words_in_text = set(re.findall(r"\b\w+\b", phrase.lower()))  # Tokenize text
    return list(words_in_text & SLANG_WORDS_SET)  # Intersection with slang set


def slang_words_def(slang_contained):
    # initialize empty dictionary to hold words and their definitions
    word_defs = dict()

    # iterate through the slang_contained list and look up the definiions of each word
    for word in slang_contained:
        definition = get_word_definition(word)
        word_defs[word] = definition

    return word_defs
