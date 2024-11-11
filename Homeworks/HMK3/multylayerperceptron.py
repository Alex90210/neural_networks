import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class MultyLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05, batch_size=50,
                 lr_patience=3, lr_decay_factor=0.5, min_lr=0.001):
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

        self.weights_i_h = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_h = np.zeros((1, hidden_size))
        self.weights_h_o = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_o = np.zeros((1, output_size))

    def forward(self, input_data):
        self.hidden_i = np.dot(input_data, self.weights_i_h) + self.bias_h
        self.hidden_o = sigmoid(self.hidden_i)

        self.output_i = np.dot(self.hidden_o, self.weights_h_o) + self.bias_o
        self.output_o = softmax(self.output_i)
        return self.output_o

    def backward(self, input_data, true_labels):
        batch_size = input_data.shape[0]
        # one hot
        true_labels_one_hot = np.zeros((batch_size, self.output_size))
        true_labels_one_hot[np.arange(batch_size), true_labels] = 1

        # gradients
        output_error = self.output_o - true_labels_one_hot
        delta_weights_h_to_o = np.dot(self.hidden_o.T, output_error) / batch_size
        delta_o_bias = np.sum(output_error, axis=0, keepdims=True) / batch_size

        hidden_error = np.dot(output_error, self.weights_h_o.T) * sigmoid_derivative(self.hidden_o)
        delta_weights_i_to_h = np.dot(input_data.T, hidden_error) / batch_size
        delta_h_bias = np.sum(hidden_error, axis=0, keepdims=True) / batch_size

        self.weights_h_o -= self.learning_rate * delta_weights_h_to_o
        self.bias_o -= self.learning_rate * delta_o_bias
        self.weights_i_h -= self.learning_rate * delta_weights_i_to_h
        self.bias_h -= self.learning_rate * delta_h_bias

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
            print(f"decreased lr from {old_lr:.6f} to {self.learning_rate:.6f}")

    def train(self, input_data, true_labels, X_val=None, y_val=None, epochs=100):
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
            
            print(f"epoch {epoch+1}, val acc: {val_accuracy:.4f}, lr: {self.learning_rate:.6f}")