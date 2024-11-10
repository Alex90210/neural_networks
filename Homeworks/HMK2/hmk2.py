import numpy as np
from torchvision.datasets import MNIST
from typing import Tuple, List
import matplotlib.pyplot as plt


# Helper functions
def download_mnist(is_train: bool) -> Tuple[List[np.ndarray], List[int]]:
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return mnist_data, mnist_labels


def normalize_data(data: List[np.ndarray]) -> np.ndarray:
    return np.array(data) / 255.0


def one_hot_encode(labels: List[int], num_classes: int = 10) -> np.ndarray:
    return np.eye(num_classes)[labels]


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]


class Perceptron:
    def __init__(self, input_size: int, output_size: int):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.W) + self.b
        return softmax(z)

    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        m = X.shape[0]
        dZ = y_pred - y_true
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        self.W -= learning_rate * dW
        self.b -= learning_rate * db


def train(model: Perceptron, X_train: np.ndarray, y_train: np.ndarray,
          X_test: np.ndarray, y_test: np.ndarray, epochs: int, batch_size: int, learning_rate: float):
    train_losses, test_accuracies = [], []

    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        # Train on batches
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # Forward pass
            y_pred = model.forward(X_batch)

            # Backward pass
            model.backward(X_batch, y_batch, y_pred, learning_rate)

        # Compute training loss
        train_pred = model.forward(X_train)
        train_loss = cross_entropy_loss(y_train, train_pred)
        train_losses.append(train_loss)

        # Compute test accuracy
        test_pred = model.forward(X_test)
        test_accuracy = np.mean(np.argmax(test_pred, axis=1) == np.argmax(y_test, axis=1))
        test_accuracies.append(test_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_accuracies


# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    train_X, train_Y = download_mnist(True)
    test_X, test_Y = download_mnist(False)

    train_X = normalize_data(train_X)
    test_X = normalize_data(test_X)

    train_Y = one_hot_encode(train_Y)
    test_Y = one_hot_encode(test_Y)

    # Initialize and train the model
    model = Perceptron(input_size=784, output_size=10)
    epochs = 100
    batch_size = 100
    learning_rate = 0.1

    train_losses, test_accuracies = train(model, train_X, train_Y, test_X, test_Y, epochs, batch_size, learning_rate)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

    print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")