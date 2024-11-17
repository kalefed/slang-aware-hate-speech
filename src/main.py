import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from tensorflow import keras
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt_tab")
nltk.download("punkt")  # At first you have to download these nltk packages.
nltk.download("stopwords")
nltk.download("wordnet")
np.random.seed(0)
import tensorflow as tf

tf.random.set_seed(1)
import joblib

import warnings

warnings.filterwarnings("ignore")


# Getting the data
train = pd.read_csv(
    "../slang-aware-hate-speech/data/covid-19-nlp-text-classification/Corona_NLP_train.csv",
    encoding="latin1",
)
test = pd.read_csv(
    "../slang-aware-hate-speech/data/covid-19-nlp-text-classification/Corona_NLP_test.csv",
    encoding="latin1",
)

df = pd.concat([train, test])
df
df.info()

df.drop(["UserName", "ScreenName", "Location", "TweetAt"], axis=1, inplace=True)

df.drop_duplicates(inplace=True)
df

sns.set_style("whitegrid")
sns.set(rc={"figure.figsize": (11, 4)})
sns.countplot(x="Sentiment", data=df)

df["Sentiment"].value_counts()

df.replace(
    ["Extremely Negative", "Extremely Positive", "Extremely Pos"],
    ["Negative", "Positive", "Positive"],
    inplace=True,
)
df["Sentiment"].value_counts()

nltk.download("wordnet")


stop_words = stopwords.words("english")  # defining stop_words
stop_words.remove(
    "not"
)  # removing not from the stop_words list as it contains value in negative movies
lemmatizer = WordNetLemmatizer()


def nlpPreprocessing(tweet):
    # Data cleaning
    tweet = re.sub(r"@\w+", "", tweet)  # Remove mentions
    tweet = re.sub(r"#\w+", "", tweet)  # Remove hashtags
    tweet = re.sub(r"https?://\S+|www\.\S+", "", tweet)  # Remove URLs
    tweet = re.sub(r"<.*?>", "", tweet)  # Remove HTML tags
    tweet = re.sub("[^A-Za-z]+", " ", tweet)  # Keep only alphabetic characters

    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenization
    tokens = nltk.word_tokenize(tweet)  # Convert text to tokens

    # Remove single-character tokens (except meaningful ones like 'i' and 'a')
    tokens = [word for word in tokens if len(word) > 1]

    # Remove stopwords
    tweet = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tweet = [lemmatizer.lemmatize(word) for word in tweet]

    # Join words back into a single string
    tweet = " ".join(tweet)

    return tweet


df["OriginalTweet"] = df["OriginalTweet"].apply(nlpPreprocessing)
df.head()
df["Sentiment"] = df["Sentiment"].apply(lambda x: x.lower())
df.head()

l = {"neutral": 0, "positive": 1, "negative": 2}
df["Sentiment"] = df["Sentiment"].map(l)

print("testing df for word cloud", df)

# WORD CLOUD
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])

df_pos = df[df["Sentiment"] == "positive"]["OriginalTweet"]
df_neg = df[df["Sentiment"] == "negative"]["OriginalTweet"]
df_neu = df[df["Sentiment"] == "neutral"]["OriginalTweet"]

words_list = df_pos.unique().tolist()
words = " ".join(words_list)


wordcloud1 = WordCloud(
    width=800,
    height=800,
    background_color="white",
    colormap="Greens",
    stopwords=stop_words,
    min_font_size=10,
).generate(words)

ax1.imshow(wordcloud1)
ax1.axis("off")
ax1.set_title("Positive Sentiment", fontsize=35)

words_list = df_neg.unique().tolist()
words = " ".join(words_list)


wordcloud2 = WordCloud(
    width=800,
    height=800,
    background_color="white",
    colormap="Reds",
    stopwords=stop_words,
    min_font_size=10,
).generate(words)
ax2.imshow(wordcloud2)
ax2.axis("off")
ax2.set_title("Negative Sentiment", fontsize=35)

words_list = df_neu.unique().tolist()
words = " ".join(words_list)


wordcloud3 = WordCloud(
    width=800,
    height=800,
    background_color="white",
    colormap="Greys",
    stopwords=stop_words,
    min_font_size=10,
).generate(words)
ax3.imshow(wordcloud3)
ax3.axis("off")
ax3.set_title("Neutal Sentiment", fontsize=35)

l = {"neutral": 0, "positive": 1, "negative": 2}
df["Sentiment"] = df["Sentiment"].map(l)
df
