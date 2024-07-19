import pandas as pd

# Read dataset, drop NA values, and concatenate title and text columns
df = pd.read_csv("data/raw/WELFake.csv").dropna()
df["text"] = df["title"] + ' ' + df["text"]

# remove title and unnamed column
df.drop(columns=["title"], inplace=True)
df.drop(columns=["Unnamed: 0"], inplace=True)

# lowercase text for uniformity
df["text"] = df["text"].apply(lambda x: x.lower())

# write to new file
df.to_csv("data/processed/WELFakeProcessed.csv", mode= "w", index=False)
