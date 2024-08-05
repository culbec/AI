import numpy as np


class LossFunction:
    def compute(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredErrorLoss(LossFunction):
    def compute(self, y_true, y_pred):
        if isinstance(y_true, list) and isinstance(y_pred, list):
            total_error = 0
            for i in range(len(y_true)):
                total_error += (y_pred[i] - y_true[i]) ** 2
                return total_error / len(y_true)
        else:
            return (y_pred - y_true) ** 2

    def derivative(self, y_true, y_pred):
        if isinstance(y_true, list) and isinstance(y_pred, list):
            return [2 * (pred - true) / len(y_true) for true, pred, in zip(y_true, y_pred)]
        else:
            return 2 * (y_pred - y_true)

class HingeLoss(LossFunction):
    def compute(self, y_true, y_pred):
        if isinstance(y_true, list) and isinstance(y_pred, list):
            error = 0.0
            for yt, yp in zip(y_true, y_pred):
                error += max(0, 1 - yp * yt)
            return error / len(y_pred)
        else:
            return max(0, 1 - yp * yt)
    def derivative(self, y_true, y_pred):
        if isinstance(y_true, list) and isinstance(y_pred, list):
            return [-true if true * pred < 1 else 0 for true, pred in zip(y_true, y_pred)]
        else:
            return -y_true if y_true * y_pred < 1 else 0