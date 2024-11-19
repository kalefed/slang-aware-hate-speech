import re
import demoji
import pandas as pd

#STEP 1 CLEAN THE TEXT FROM UNECESARY DATA
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # Remove hashtags
    return text

# Load emoji mapping data
demoji.download_codes()
# STEP 2 CONVERT ALL EMOJIS
def convert_emojis_to_text(text):
    return demoji.replace(text, sep=" ")

# STEP 3 GET SLANG MEANINGS 
def slang_meanings(text):
    pass

# STEP 4 RUN EVERYTHING FOR PREPROCESSING
def preprocess_pipeline(text):
    text = clean_text(text) 
    text = convert_emojis_to_text(text)  
    text = slang_meanings(text) # Do stuff with the slang here
    return text

# STEP 5 TOKENIZE EVERYTHING
def tokenizer(text):
    #happens after preprocessing - use the pretrained bert tokenizer?
    pass

# SAVE ALL THE DATA
def save_to_tsv(data, file_path):
    df = pd.DataFrame(data, columns=['label', 'text'])
    df.to_csv(file_path, sep='\t', index=False)

# Example - but depends on how kj formats this originally
dataset = [
    (0, "This is a neutral comment."),
    (1, "This is a racist comment."),
    (2, "This is a sexist comment."),
    (3, "This is a homophobic comment."),
]


processed_text = preprocess_pipeline(unprocessed_text)
tokenized_text = tokenizer(processed_text)
save_to_tsv(tokenized_text, './data/processed_data.tsv')



