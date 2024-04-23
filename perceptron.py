import numpy as np

class Perceptron():

    def __init__(self, N, epochs=10, lr=0.1) -> None:
        self.epochs = epochs
        self.learning_rate = lr
        self.W = np.random.randn(N)

    def step(self, X):
        X = np.where(X >= 0, 1, 0)
        return X
        
    def fit(self, X, y):

        losses = []

        for epoch in range(self.epochs):
            p = self.step(np.dot(X, self.W))
            errors = p - y
            loss = np.sum(errors ** 2) / 2
            losses.append(loss)

            self.W += -self.learning_rate * np.dot(X.T, errors)
        print(f'losses: {losses}')

    def predict(self, X):
        return self.step(np.dot(X, self.W))