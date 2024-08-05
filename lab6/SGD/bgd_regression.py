from statistics import mean
import numpy as np


# Processes the whole data set at each run.
# Computationally expensive.
class BGDRegression:
    def __init__(self, learning_rate=0.01, no_epochs=1000):
        self.intercept_ = []
        self.coef_ = np.array([])

        self.learning_rate = learning_rate
        self.no_epochs = no_epochs

    def fit(self, x, y):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]
        for _ in range(self.no_epochs):
            y_errors = []

            # Computing the error of the dataset.
            for i in range(len(x)):
                y_pred = self.eval(x[i])
                y_error = y_pred - y[i]
                y_errors.append(y_error)

            # Determining the mean of the error set.
            y_error = mean(y_errors)

            # Readjusting the coefficients.
            for i in range(len(x)):
                for j in range(0, len(x[0])):
                    self.coef_[j] -= self.learning_rate * y_error * x[i][j]
                self.coef_[-1] -= self.learning_rate * y_error

        self.intercept_ = [self.coef_[-1]]
        self.coef_ = np.array(self.coef_[:-1], ndmin=1)

    def eval(self, xi):
        # Evaluating the intercept.
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, x):
        # Predicting the values of the dataset.
        y_pred = [self.eval(xi) for xi in x]
        return y_pred
