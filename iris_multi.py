import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from multi_perceptron import Perceptron


data = pd.read_csv('Iris.csv')
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# bias column
bias_col = np.ones((X.shape[0], 1))
X = np.hstack((X, bias_col))

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

P = Perceptron(X_train.shape[1], lr=0.01, epochs=10)
P.fit(X_train, y_train)

predictions = P.predict(X_test)
for pred, target in zip(predictions, y_test):
    print(f'Predicted: {pred}     Ground truth: {target}')