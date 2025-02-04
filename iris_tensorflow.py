import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

print("[INFO] GPU available: ", tf.config.list_physical_devices('GPU'))

# Model architecture
Model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(4,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')  
])

Model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training loop
Model.fit(X_train, y_train, epochs=100, batch_size=None, verbose=1)

# Accuracy
loss, accuracy = Model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy:.4f}')