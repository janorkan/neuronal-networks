# Neuronal Network Library
# 3-layer feedforward neuronal network with backpropagation

import numpy as np
from scipy import special


class NN3Classifier:
    def __init__(
        self,
        input_layer,
        hidden_layers,
        output_layer,
        learning_rate,
        epochs,
        random_seed,
    ):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.w1 = np.random.uniform(-0.5, 0.5, (self.hidden_layers, self.input_layer))
        self.w2 = np.random.uniform(-0.5, 0.5, (self.output_layer, self.hidden_layers))

    def predict(self, X, w_1, w_2):
        input_vector = np.array(X, ndmin=2).T

        X_1 = np.dot(w_1, input_vector)
        O_1 = special.expit(X_1)

        X_2 = np.dot(w_2, O_1)
        O_2 = special.expit(X_2)

        return O_1, O_2

    def fit(self, X, y):
        for i in range(self.epochs):
            w1, w2 = self.train(X, y)
            if i == self.epochs - 1:
                print("Training over!")
        return w1, w2

    def train(self, X, y):
        input_vector = np.array(X, ndmin=2).T
        targets = np.array(y, ndmin=2).T

        X_1 = np.dot(self.w1, input_vector)
        O_1 = special.expit(X_1)

        X_2 = np.dot(self.w2, O_1)
        O_2 = special.expit(X_2)

        error_output = targets - O_2
        error_hidden = np.dot(self.w2.T, error_output)

        self.w1 += self.learning_rate * np.dot(
            (error_hidden * O_1 * (1 - O_1)), np.transpose(input_vector)
        )
        self.w2 += self.learning_rate * np.dot(
            (error_output * O_2 * (1 - O_2)), np.transpose(O_1)
        )

        return self.w1, self.w2

    def performance(self, y_test, y_predicted):
        self.y_test = y_test
        self.y_predicted = y_predicted

        correct = []
        for i in range(0, len(y_test)):
            if np.argmax(y_predicted[i]) == np.argmax(y_test[i]):
                correct.append(i)
        performance = len(correct) / len(y_test)
        return performance, correct
