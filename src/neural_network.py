import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

device = (
        "cuda" 
        if torch.cuda.is_available() 
        else "mps"
        if torch.backends.mps.is_available() 
        else "cpu"
        )

print(f"Using {device} device")

# define neural network
class NeuralNetwork(nn.Module):
    def __init__(self, in_features=300, h1=128, h2=128, out_features=1):
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

# read dataset
df = pd.read_csv("data/processed/WELFakeProcessed.csv")

# split data into X and Y
x = df["text"]
y = df["label"].values

# convert text to vectors
vectorizer = TfidfVectorizer(max_features=300)
x = vectorizer.fit_transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

criterion = nn.BCEWithLogitsLoss() # binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# train model
epochs = 500
losses = []
for i in range(epochs):
    y_pred = model.forward(x_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().cpu().numpy())

    print(f"Epoch {i + 1} with loss {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plot loss
plt.plot(range(epochs), losses)
plt.ylabel("Error/Loss")
plt.xlabel("Epoch")
plt.show()

# evaluate model
with torch.no_grad():
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)
    print(loss)

# test model
correct = 0
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = model.forward(data)

        print(f"Count {i + 1}, {str(y_val)} \t {y_test[i]}")

        if y_val.argmax().item == y_val:
            correct += 1

        if i == 1000:
            break
    
print("We got " + str(correct) + " right!")

# TODO: model is not doing well on validation data, need to find discrepancies in the code/data
