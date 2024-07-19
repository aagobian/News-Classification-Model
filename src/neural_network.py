import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader 

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
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(300, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
