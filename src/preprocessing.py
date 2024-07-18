from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re
from tqdm import tqdm

def preprocessing():
    # Read data
    df = pd.read_csv("data/raw/WELFake.csv")

    # Drop NA values
    df = df.dropna()

    # Initialize tqdm for monitoring progress
    tqdm.pandas()

    # Process title and text columns
    for col in ["title", "text"]:
        # Apply tqdm on pandas series
        tqdm.pandas(desc=f"Processing {col} column...")
        
        print(df.head())
        print("Tokenizing sentences...")

        # Perform segmentation
        df[col] = df[col].progress_apply(sent_tokenize)

        print(df.head())
        print("Removing punctuation...")

        # Remove punctuation
        df[col] = df[col].progress_apply(lambda x: [re.sub(r"[^a-zA-Z0-9]", " ", sentence) for sentence in x])
        
        print(df.head())
        print("Tokenizing words...")

        # Tokenize words
        df[col] = df[col].progress_apply(lambda x: [word_tokenize(sentence) for sentence in x])
        
        print(df.head())
        print("Removing stop words...")
        
        # Remove stop words
        stop_words = set(stopwords.words("english"))
        df[col] = df[col].progress_apply(lambda x: [[word for word in sentence if word.lower() not in stop_words] for sentence in x])
        
        print(df.head())
        print("Lemmatizing words...")

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        df[col] = df[col].progress_apply(lambda x: [[lemmatizer.lemmatize(word) for word in sentence] for sentence in x])
        
        print(df.head())
        
    # Save preprocessed data
    print("Saving processed data...")
    df.to_csv("data/processed/WELFakeProcessed.csv", mode="w", index=False)

if __name__ == "__main__":
    preprocessing()