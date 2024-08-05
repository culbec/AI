import numpy as np
from myActivationFunctions import *

class NetworkLayer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class DenseLayer(NetworkLayer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights = (np.random.rand(input_size, output_size) - 0.5).tolist()
        self.bias = (np.random.rand(output_size) - 0.5).tolist();
        self.deltas = None
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = [0] * len(self.weights[0])
 
        for i in range(len(input_data)):
            for j in range(len(self.weights[0])):
                self.output[j] += input_data[i] * self.weights[i][j]
                
        for i in range(len(self.output)):
            self.output[i] += self.bias[i]
        
        # Normalizing the output.
        sigmoid = SigmoidActivation()
        
        self.output = sigmoid.activate(self.output)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        # Readjusting the neurons based on the previous computations.
        self.deltas = [sum(w*d for w, d in zip(weight_row, output_error)) for weight_row in self.weights]

        for j in range(len(self.weights[0])):
            for i in range(len(self.input)):
                self.weights[i][j] -= learning_rate * output_error[j] * self.input[i]

        for i in range(len(self.bias)):
            self.bias[i] -= learning_rate * output_error[i]

        return self.deltas
        