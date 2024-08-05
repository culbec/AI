import numpy as np
from myLossFunctions import *
from myActivationFunctions import *
from myNetworkLayers import *


class MyANN:
    def __init__(self, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = []
        self.loss_function = MeanSquaredErrorLoss()
        self.activation_function = SigmoidActivation()

    def use(self, loss_f: LossFunction, activation_f: ActivationFunction):
        """
            Specifies other loss and activation functions for the network.
            
            :param loss_f: The loss function.
            :param activation_f: The activation function.
        """
        self.loss_function = loss_f
        self.activation_function = activation_f

    def add(self, layer: NetworkLayer):
        """
            Adds a layer to the network.
            :param layer: Layer to be added.
        """
        self.layers.append(layer)

    def compute_loss(self, y_true: list | float, y_pred: list | float):
        """
            Computes the loss of a prediction.
            :param y_true: The ground truth.
            :param y_pred: The predicted value:
            
            :rtype: float
            :return: The loss.
        """
        return self.loss_function.compute(y_true, y_pred)

    def __to_categorical(self, y):
        max_lab = max(y)
        y = [[0] * label + [1] + [0] * (max_lab - label) for label in y]
        return y

    def __forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward_propagation(input_data)
        return input_data

    def __backward(self, y_true, learning_rate):
        # Retrieving the output from the last neuron (the output neuron).
        y_pred = self.layers[-1].output

        # Determining the actual loss.
        loss = self.loss_function.compute(y_true, y_pred)

        # Determining the error.
        output_error = [2 * (y_pred[i] - y_true[i]) * y_pred[i] * (1 - y_pred[i]) for i in range(len(y_true))]
        for layer in reversed(self.layers):
            output_error = layer.backward_propagation(output_error, learning_rate)

        return loss

    def predict(self, input_data: list):
        # If input_data is a 1D array, reshape it to a 2D array with 1 row
        if np.ndim(input_data) == 1:
            input_data = np.reshape(input_data, (1, -1))

        samples = len(input_data)
        predictions = []

        # Run network over all samples.
        for i in range(samples):
            output = self.__forward(input_data[i])
            label = output.index(max(output))
            predictions.append(label)

        return np.array(predictions)

    def fit(self, X_train, y_train):
        samples = len(X_train)
        y_categorical = self.__to_categorical(y_train)

        for i in range(self.epochs):
            total_loss = 0
            for j in range(samples):
                _ = self.__forward(X_train[j])
                total_loss += self.__backward(y_categorical[j], self.learning_rate)

            avg_loss = total_loss / samples
            print(f'Epoch {i + 1} | Loss = {avg_loss}')
