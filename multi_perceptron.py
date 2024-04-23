import numpy as np

def sigmoid_activation(X):
    return 1 / (1 + np.exp(-X))

def threshold(X):
    return np.where(X >= 0.5, 1, 0)

class Perceptron:

    def __init__(self, N, epochs=10, lr = 0.01) -> None:
        self.W = np.random.randn(N)
        self.epochs = epochs
        self.learning_rate = lr

    def forward(self, X):
        activation = np.dot(X, self.W)
        return sigmoid_activation(activation)
    
    def fit(self, X, y):
        for epoch in range(self.epochs):
            for input, label in zip(X, y):
                pred = self.forward(input)
                error = label - pred
                self.W += -self.learning_rate * error * input

    def predict(self, X):
        preds = []
        for input in X:
            pred = self.forward(input)
            preds.append(threshold(pred))

        return np.array(preds)