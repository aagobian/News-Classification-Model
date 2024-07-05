import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

def preprocessing():
    # specify file path and read data
    file_path = "data/raw/WELFake.csv"
    df = pd.read_csv(file_path)

    # drop na values
    df = df.dropna()

    # convert relevant columns to lowercase
    df["title"] = df["title"].str.lower()
    df["text"] = df["text"].str.lower()

    # tokenize relevant columns
    df["title"] = df["title"].apply(word_tokenize)
    df["text"] = df["text"].apply(word_tokenize)

    # TODO: complete prepreprocessing steps here

if __name__ == "__main__":
    preprocessing()