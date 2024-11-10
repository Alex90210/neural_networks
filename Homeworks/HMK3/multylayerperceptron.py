import numpy as np
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class MultyLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05, batch_size=50,
                 lr_patience=3, lr_decay_factor=0.5, min_lr=1e-6):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.lr_patience = lr_patience
        self.lr_decay_factor = lr_decay_factor 
        self.min_lr = min_lr
        self.best_val_accuracy = 0
        self.patience_counter = 0

        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def forward(self, input_data):
        # hidden layer
        self.hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output_output = softmax(self.output_input)
        return self.output_output

    def backward(self, input_data, true_labels):
        batch_size = input_data.shape[0]
        # correct labels one hot encoded
        true_labels_one_hot = np.zeros((batch_size, self.output_size))
        true_labels_one_hot[np.arange(batch_size), true_labels] = 1

        output_error = self.output_output - true_labels_one_hot

        # gradients for output layer
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error) / batch_size
        d_bias_output = np.sum(output_error, axis=0, keepdims=True) / batch_size

        # error for hidden layer
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)

        # gradients for hidden layer
        d_weights_input_hidden = np.dot(input_data.T, hidden_error) / batch_size
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True) / batch_size

        self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
        self.bias_output -= self.learning_rate * d_bias_output
        self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden

    def adjust_learning_rate(self, val_accuracy):
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.lr_patience:
            old_lr = self.learning_rate
            self.learning_rate = max(self.learning_rate * self.lr_decay_factor, self.min_lr)
            self.patience_counter = 0
            
            if old_lr != self.learning_rate:
                print(f"decreased lr from {old_lr:.6f} to {self.learning_rate:.6f}")

    def train(self, input_data, true_labels, X_val=None, y_val=None, epochs=100, target_accuracy=0.95):
        history = {'train_accuracy': [], 'val_accuracy': [], 'learning_rates': []}
        
        for epoch in range(epochs):
            permutation = np.random.permutation(input_data.shape[0])
            input_data = input_data[permutation]
            true_labels = true_labels[permutation]

            for i in range(0, input_data.shape[0], self.batch_size):
                x_batch = input_data[i:i+self.batch_size]
                y_batch = true_labels[i:i+self.batch_size]
                self.forward(x_batch)
                self.backward(x_batch, y_batch)

            val_outputs = self.forward(X_val)
            val_predictions = np.argmax(val_outputs, axis=1)
            val_accuracy = np.mean(val_predictions == y_val)
            
            self.adjust_learning_rate(val_accuracy)
            
            history['val_accuracy'].append(val_accuracy)
            history['learning_rates'].append(self.learning_rate)
            
            print(f"epoch {epoch+1}/{epochs}, val acc: {val_accuracy:.4f}, lr: {self.learning_rate:.6f}")

        return history