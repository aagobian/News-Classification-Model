import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

while True:
    try:
        file_path = input("Enter the path to the dataset: ")
        df = pd.read_csv(file_path)
        break
    except FileNotFoundError:
        print("File not found. Ensure the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Load the model architecture
class NeuralNetwork(nn.Module):
    def __init__(self, in_features=1000, h1=800, h2=600, h3=400, h4=200, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1) # Input layer
        self.fc2 = nn.Linear(h1, h2) # Hidden layer
        self.fc3 = nn.Linear(h2, h3) # Hidden layer
        self.fc4 = nn.Linear(h3, h4) # Hidden layer
        self.out = nn.Linear(h4, out_features) # Output layer
        
    # Forward pass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x) 
        return x
    
model = NeuralNetwork()
model.load_state_dict(torch.load("models/neural_network_weights.pth"))
model.eval()

# Initialize vectorizer
vectorizer = TfidfVectorizer(max_features=1000)  # Ensure the same parameters as used during training
x = vectorizer.fit_transform(df["text"]).toarray()
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(df["label"].values, dtype=torch.float32).unsqueeze(1)

correct = 0
total = len(df)
with torch.no_grad():
    for i, data in enumerate(df):
        data = x[i].unsqueeze(0)  # Add batch dimension
        label = y[i].unsqueeze(0)  # Add batch dimension
        
        y_val = model(data)  # Pass data through the model
        y_val = torch.sigmoid(y_val)  # Convert to probability
        predicted = (y_val > 0.5).float()  # Convert to binary

        if predicted.eq(label).all().item():  # Check if prediction is correct
            correct += 1

# Print accuracy
print(f"Accuracy: {round(100 * correct / total, 2)}%")
print(f"That's {correct} out of {total} right!")


# TODO: Fix model not predicting correctly