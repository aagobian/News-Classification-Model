from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pandas as pd

# read dataset
df = pd.read_csv("data/processed/WELFakeProcessed.csv")

# split data into X and Y
X = df["text"]
Y = df["label"]

# split data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=15)

# create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words="english", max_df=0.7)), # initialize TF-IDF vectorizer
    ('nb', MultinomialNB()) # create and fit Naive Bayes model
])

# fit pipeline on training data
pipeline.fit(X_train, Y_train)

# predict on testing data
nb_pred = pipeline.predict(X_test)

# print classification report and confusion matrix
print("Naive Bayes Classification Report:")
print(classification_report(Y_test, nb_pred))

print("Naive Bayes Confusion Matrix:")
print(confusion_matrix(Y_test, nb_pred))
