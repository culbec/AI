import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, no_epochs=1000, threshold=0.5):
        self.intercept_ = []
        self.coef_ = np.array([])
        self.loss = []
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.no_epochs = no_epochs

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]

        for _ in range(self.no_epochs):
            epoch_loss = []
            for i in range(len(x)):
                y_pred = self.__sigmoid(self.eval(x[i]))
                y_error = y_pred - y[i]
                epoch_loss.append(y_error)

                for j in range(0, len(x[0])):
                    self.coef_[j] -= self.learning_rate * y_error * x[i][j]
                self.coef_[-1] -= self.learning_rate * y_error
            self.loss.append(np.mean(epoch_loss))

        self.intercept_ = [self.coef_[-1]]
        self.coef_ = np.array(self.coef_[:-1], ndmin=1)

    def eval(self, xi, ):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict_proba(self, x):
        y_pred = [[self.__sigmoid(el) for el in xi] for xi in x]
        return y_pred
