import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# read dataset
df = pd.read_csv("data/processed/WELFakeProcessed.csv")

# split data into X and Y
X = df["text"]
Y = df["label"]

device = (
        "cuda" 
        if torch.cuda.is_available() 
        else "mps"
        if torch.backends.mps.is_available() 
        else "cpu"
        )

print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, in_features=300, h1=128, h2=128, out_features=1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(300, 128),
        #     nn.ReLU(),
        #     nn.Linear(300, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        #     nn.ReLU()
        # )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
model = NeuralNetwork().to(device)
print(model)
