import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Check if GPU is available
device = (
        "cuda" 
        if torch.cuda.is_available() 
        else "mps"
        if torch.backends.mps.is_available() 
        else "cpu"
        )

print(f"Using {device} device")

# Read dataset
df = pd.read_csv("data/processed/WELFakeProcessed.csv")

# Convert text to vectors
vectorizer = TfidfVectorizer(max_features=1000)  

x = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"].values

# Define neural network
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
    
# Create model
model = NeuralNetwork().to(device)
print(model)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

criterion = nn.BCEWithLogitsLoss() # Binary cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001) # Adam optimizer with L2 regularization

# Train model
epochs = 200
losses = []
for i in range(epochs):
    model.train() # Set model to training mode
    optimizer.zero_grad() # Zero gradients
    y_pred = model.forward(x_train) # Get predictions
    
    loss = criterion(y_pred, y_train) # Calculate loss
    losses.append(loss.detach().cpu().numpy()) # Store loss

    loss.backward() # Backward pass
    optimizer.step() # Update weights

    if (i + 1) % 10 == 0:  # Print every 10 epochs
        print(f"Epoch {i + 1} with loss {loss}")

# Plot loss
plt.plot(range(epochs), losses)
plt.ylabel("Error/Loss")
plt.xlabel("Epoch")
plt.show()

# Evaluate model
model.eval() # Set model to evaluation mode
with torch.no_grad():
    y_eval = model(x_test) # Get predictions
    loss = criterion(y_eval, y_test) # Calculate loss
    print(f"Evaluation loss: {loss.item()}") 

# Test model
correct = 0
total = len(y_test)
with torch.no_grad():
    for i, data in enumerate(x_test):
        y_val = model(data.unsqueeze(0)) # Unsqueeze to add batch size of 1  
        y_val = torch.sigmoid(y_val)  # Convert to probability
        predicted = (y_val > 0.5).float()  # Convert to binary

        if predicted.eq(y_test[i]).all().item():  # Check if prediction is correct
            correct += 1

# Print accuracy
print(f"Accuracy: {round(100 * correct / total, 2)}%")
print(f"That's {correct} out of {total} right!")

# Save model
while True:
    save = input("Do you want to save the model? It will overwrite the saved model if it exists. (y/n): ")
    if save.lower() == "y":
        torch.save(model.state_dict(), "models/neural_network_weights.pth")
        break
    elif save.lower() == "n":
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
