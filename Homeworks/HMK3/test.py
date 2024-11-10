import numpy as np
from typing import Tuple

# Define the sigmoid and softmax functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Define the MultyLayerPerceptron class
class MultyLayerPerceptron:
    def __init__(self, input_size: int, hidden_layer_size: int, output_size: int):
        self.weights_input_hidden = np.random.normal(0, np.sqrt(2.0 / (input_size + hidden_layer_size)),
                                                     (input_size, hidden_layer_size))
        self.bias_hidden = np.zeros((1, hidden_layer_size))
        self.weights_hidden_output = np.random.normal(0, np.sqrt(2.0 / (hidden_layer_size + output_size)),
                                                      (hidden_layer_size, output_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.hidden_layer_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = softmax(self.output_layer_input)

        return self.hidden_layer_input, self.hidden_layer_output, self.output_layer_input, self.output_layer_output

# Example parameters
input_size = 10
hidden_layer_size = 5
output_size = 3
batch_size = 4

# Create a random input batch
input_data = np.random.rand(batch_size, input_size) - 0.5  # Preprocessed input

# Initialize the MultyLayerPerceptron model
mlp = MultyLayerPerceptron(input_size, hidden_layer_size, output_size)

# Perform forward propagation
hidden_layer_input, hidden_layer_output, output_layer_input, output_layer_output = mlp.forward(input_data)

# Print the results
print("Input Data:")
print(input_data)
print("\nHidden Layer Input:")
print(hidden_layer_input)
print("\nHidden Layer Output:")
print(hidden_layer_output)
print("\nOutput Layer Input:")
print(output_layer_input)
print("\nOutput Layer Output:")
print(output_layer_output)