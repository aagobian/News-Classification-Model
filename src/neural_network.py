import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

device = (
        "cuda" 
        if torch.cuda.is_available() 
        else "mps"
        if torch.backends.mps.is_available() 
        else "cpu"
        )

print(f"Using {device} device")

# read dataset
df = pd.read_csv("data/processed/WELFakeProcessed.csv")

# convert text to vectors
vectorizer = TfidfVectorizer(max_features=300)  

x = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"].values

# define neural network
class NeuralNetwork(nn.Module):
    def __init__(self, in_features=300, h1=128, h2=64, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1) # input layer
        self.fc2 = nn.Linear(h1, h2) # hidden layer
        self.out = nn.Linear(h2, out_features) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) 
        x = self.out(x)
        return x
    
# create model and print it
model = NeuralNetwork().to(device)
print(model)

# split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

criterion = nn.BCEWithLogitsLoss() # binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

# train model
epochs = 1000
losses = []
for i in range(epochs):
    y_pred = model.forward(x_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().cpu().numpy())

    if (i + 1) % 100 == 0:  # Print every 100 epochs
        print(f"Epoch {i + 1} with loss {loss}")

    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot loss
plt.plot(range(epochs), losses)
plt.ylabel("Error/Loss")
plt.xlabel("Epoch")
plt.show()

# Evaluate model
with torch.no_grad():
    y_eval = model(x_test) # get predictions
    loss = criterion(y_eval, y_test) # calculate loss
    print(f"Evaluation loss: {loss.item()}") 

# Test model
correct = 0
total = len(y_test)
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = model(data.unsqueeze(0)) # unsqueeze to add batch size of 1  
        y_val = torch.sigmoid(y_val)  # convert to probability
        predicted = (y_val > 0.5).float()  # convert to binary

        if predicted.eq(y_test[i]).all().item():  # check if prediction is correct
            correct += 1

print(f"Accuracy: {round(100 * correct / total, 2)}%")
print(f"That's {correct} out of {total} right!")

# TODO: Model is overfitting, try to improve it (maybe dropout or L2 regularization)
