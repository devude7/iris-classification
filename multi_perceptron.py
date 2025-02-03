import numpy as np

def sigmoid_activation(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    return X * (1 - X)

def relu_activation(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return np.where(X > 0, 1, 0)

def softmax_activation(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

class MultiPerceptron:

    def __init__(self, layers, activation="relu", epochs=10, lr=0.01) -> None:
        self.W = []
        
        # Weights initialization
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # Output layer with no bias
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

        self.layers = layers
        self.epochs = epochs
        self.learning_rate = lr

        # Activation functions and derivatives
        self.activation_name = activation
        self.activations = {
            "sigmoid": sigmoid_activation,
            "relu": relu_activation,
            "softmax": softmax_activation
        }
        self.derivatives = {
            "sigmoid": sigmoid_derivative,
            "relu": relu_derivative
        }

    def fit(self, X, y):
        X = np.c_[X, np.ones((X.shape[0], 1))]  # bias

        for epoch in range(self.epochs):
            for x, target in zip(X, y):
                self.backpropagation(x, target)

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                loss = self.loss(X, y)
                print(f"[INFO] epoch={epoch + 1}, loss={loss:.4f}")

    def backpropagation(self, x, y):
        Ac = [np.atleast_2d(x)]

        # Forward pass
        for i in range(len(self.W) - 1):
            activation = np.dot(Ac[i], self.W[i])
            Ac.append(self.activations[self.activation_name](activation))

        # Output layer - softmax
        activation = np.dot(Ac[-1], self.W[-1])
        output = softmax_activation(activation)
        Ac.append(output)

        error = Ac[-1] - y

        # Backpropagation
        D = [error]  

        for layer in range(len(Ac) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T) * self.derivatives[self.activation_name](Ac[layer])
            D.append(delta)

        D = D[::-1]

        # Weight update
        for layer in range(len(self.W)):
            self.W[layer] -= self.learning_rate * Ac[layer].T.dot(D[layer])

    def loss(self, X, y):
        predictions = self.predict(X, bias=False)
        return -np.sum(y * np.log(predictions + 1e-9)) / len(y)  # Cross-entropy loss

    def predict(self, X, bias=True):

        if bias:
            X = np.c_[X, np.ones((X.shape[0], 1))]

        for layer in range(len(self.W) - 1):
            X = self.activations[self.activation_name](np.dot(X, self.W[layer]))

        return softmax_activation(np.dot(X, self.W[-1]))