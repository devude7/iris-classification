import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Iris.csv')
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

X = np.array(X)
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Set device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform nparrays to torch.tensor
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Model architecture
Model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad() # Reset gradients
    outputs = Model(X_train) # Forward pass
    loss = criterion(outputs, y_train) # Calculate loss with CrossEntropy
    loss.backward() # Backpropagation - compute gradients 
    optimizer.step() # Weight update based on gradients and optimizer

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{100}, Loss: {loss.item():.4f}')

# Accuracy
with torch.no_grad():
    test_outputs = Model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')