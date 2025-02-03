import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from multi_perceptron import MultiPerceptron


data = pd.read_csv('Iris.csv')
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

X = np.array(X)
y = LabelEncoder().fit_transform(y)

y = np.eye(3)[y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

MLP = MultiPerceptron([4,16,8,3], lr=0.01, epochs=100)
MLP.fit(X_train, y_train)

predictions = MLP.predict(X_test)

pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

for pred, target, probs in zip(pred_labels, true_labels, predictions):
    percentages = [f"{p*100:.2f}%" for p in probs]  
    confidence = probs[pred] * 100  
    print(f'Predicted: {pred} ({confidence:.2f}%)     Ground truth: {target}     Probabilities: {percentages}')