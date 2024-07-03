import nltk
from nltk.tokenize import sent_tokenize
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
    nltk.download('punkt')
    df["title"] = sent_tokenize(df["title"], language="english")
    df["text"] = sent_tokenize(df["text"], language="english")

    # TODO: fix tokenization to work with pandas series
    # TODO: complete prepreprocessing steps here
