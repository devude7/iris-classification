import numpy as np

def sigmoid_activation(X):
    return 1 / (1 + np.exp(-X))

def relu_activation(X):
    return np.maximum(0, X)

def softmax_activation(X):
    exp_X = np.exp(X - np.max(X, axis=0, keepdims=True))  
    return exp_X / np.sum(exp_X, axis=0, keepdims=True)

def threshold(X):
    return np.where(X >= 0.5, 1, 0)

class MultiPerceptron:

    def __init__(self, N, epochs=10, lr = 0.01, layers = [4,3,3,3]) -> None:
        
        self.W = []

        for layer in layers[1:-1]:
            self.W.append(np.random.randn(N))

        self.epochs = epochs
        self.learning_rate = lr

    def forward(self, X, F="sigmoid"):
        activation = np.dot(X, self.W)

        activations = {
            "sigmoid": sigmoid_activation,
            "relu": relu_activation,
            "softmax": softmax_activation
        }
        
        if F not in activations:
            raise ValueError(f"Unknown activation function, use: sigmoid, relu, softmax")
        
        return activations[F](activation)
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass