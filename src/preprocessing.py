import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
import pandas as pd
import re

def preprocessing():
    # specify file path and read data
    file_path = "data/raw/WELFake.csv"
    df = pd.read_csv(file_path)

    # drop na values
    df = df.dropna()

    # convert relevant columns to lowercase
    df["title"] = df["title"].str.lower()
    df["text"] = df["text"].str.lower()

    # perform segmentation on relevant columns
    nltk.download("punkt")
    df["title"] = df["title"].apply(sent_tokenize)
    df["text"] = df["text"].apply(sent_tokenize)

    # remove punctuation from relevant columns
    df["title"] = df["title"].apply(lambda x: [re.sub(r"[^a-zA-Z0-9]", " ", sentence) for sentence in x])
    df["text"] = df["text"].apply(lambda x: [re.sub(r"[^a-zA-Z0-9]", " ", sentence) for sentence in x])

    # perform word tokenization on relevant columns
    df["title"] = df["title"].apply(lambda x: [word_tokenize(sentence) for sentence in x])
    df["text"] = df["text"].apply(lambda x: [word_tokenize(sentence) for sentence in x])

    print(df.head())

    # remove stop words from relevant columns
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    df["title"] = df["title"].apply(lambda x: [[word for word in sentence if word not in stop_words] for sentence in x])
    df["text"] = df["text"].apply(lambda x: [[word for word in sentence if word not in stop_words] for sentence in x])

    print(df.head())
    # TODO: complete prepreprocessing steps here (stop word removal, stemming, lemmatization, POS tagging, etc.)

if __name__ == "__main__":
    preprocessing()