from perceptron import Perceptron
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('Iris.csv')
X = data.iloc[:100, 1:-1]
y = data.iloc[:100, -1]
y = np.where(y == "Iris-setosa", 1, 0)

X = np.hstack((X, np.ones((X.shape[0], 1))))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25)
P = Perceptron(X.shape[1], epochs=20)
P.fit(X_train, y_train)

for input, label in zip(X_test, y_test):
    print(f'Predicted: {P.predict(input)}    Ground Truth: {label}')
print(f'Sum of errors: {np.sum(y_test - P.predict(X_test))}')
