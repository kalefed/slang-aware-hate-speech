import requests


def query_urban_dictionary(word):
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
