from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from mnist_loader import download_mnist
from multylayerperceptron import MultyLayerPerceptron

def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = X.astype('float32')
    X = X / 255.0
    X = X - 0.5
    return X, y

X_train, y_train = download_mnist(is_train=True)
X_test, y_test = download_mnist(is_train=False)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)

input_size = X_train.shape[1] 
hidden_size = 100               
output_size = 10           

mlp = MultyLayerPerceptron(input_size, hidden_size, output_size)
history = mlp.train(X_train, y_train, X_val, y_val)