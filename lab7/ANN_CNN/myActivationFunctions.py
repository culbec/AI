import numpy as np


class ActivationFunction:
    def activate(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class SigmoidActivation(ActivationFunction):
    def activate(self, x):
        if isinstance(x, list):
            return [1.0 / (1 + np.exp(-xi)) for xi in x]
        else:
            return 1.0 / (1 + np.exp(-x))

    def derivative(self, x):
        if isinstance(x, list):
            return [(self.activate(xi) * (1 - self.activate(xi))) for xi in x]
        else:
            return self.activate(x) * (1 - self.activate(x))


class ReLUActivation(ActivationFunction):
    def activate(self, x):
        if isinstance(x, list):
            return [max(0, xi) for xi in x]
        else:
            return max(0, x)

    def derivative(self, x):
        if isinstance(x, list):
            return [1 if xi > 0 else 0 for xi in x]
        else:
            return 1 if x > 0 else 0
