from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# read dataset
df = pd.read_csv("data/processed/WELFakeProcessed.csv")

# split data into X and Y
X = df["text"]
Y = df["label"]

# split data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=15)

# initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)

# fit TF-IDF on training data
train_tfidf = tfidf.fit_transform(X_train)

# transform test data using the fitted TF-IDF vectorizer
test_tfidf = tfidf.transform(X_test)

# create and fit Naive Bayes model
nb = MultinomialNB()
nb.fit(train_tfidf, Y_train)

# predict on test data
nb_pred = nb.predict(test_tfidf)

# print classification report and confusion matrix
print("Naive Bayes Classification Report:")
print(classification_report(Y_test, nb_pred))

print("Naive Bayes Confusion Matrix:")
print(confusion_matrix(Y_test, nb_pred))
