import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import sigmoid_activation, sigmoid_derivative, predict


data = pd.read_csv('iris.csv')
X = data.drop('Id', axis=1)

X = X.iloc[:100, :-1]
y = data.iloc[:100, -1]
y = np.where(y == "Iris-setosa", 1, 0)

# bias column
bias_col = np.ones((X.shape[0], 1))
X = np.hstack((X, bias_col))

X_train, X_test, y_train, y_test = train_test_split(X, y)

# hyperparameters
epochs = 50
lr = 0.01

W = np.random.randn(X.shape[1])
losses = []

for epoch in range(epochs):
    
    # activation
    predictions = predict(X_train, W)
    error = predictions - y_train

    # loss
    loss = np.sum(error**2)
    losses.append(loss)

    # gradient descent
    d = error * sigmoid_derivative(predictions)
    gradient = np.dot(X_train.T, d)

    # update weights
    W += -lr * gradient

print(losses)
