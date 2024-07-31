import pandas as pd
import torch
from create_neural_network import NeuralNetwork

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
model = NeuralNetwork()
model.load_state_dict(torch.load("models/neural_network_weights.pth"))
model.eval()

correct = 0
total = len(df)
with torch.no_grad():
    for i, data in enumerate(df):
        y_val = model(data.unsqueeze(0)) # Unsqueeze to add batch size of 1  
        y_val = torch.sigmoid(y_val)  # Convert to probability
        predicted = (y_val > 0.5).float()  # Convert to binary

        if predicted.eq(df[i]).all().item():  # Check if prediction is correct
            correct += 1

# Print accuracy
print(f"Accuracy: {round(100 * correct / total, 2)}%")
print(f"That's {correct} out of {total} right!")
