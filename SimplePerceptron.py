
import numpy as np
import matplotlib.pyplot as plb

class Perceptron:
    # initializacion with random weights
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = np.random.randn(n_inputs)
        self.output = 0
    # lineal regression
    def propagation(self, inputs):
        self.output = 1 * (self.weights.dot(inputs) > 0)
        self.inputs = inputs
    # update the weights with the lineal regression values
    def update(self, alpha, output):
        for i in range(0, self.n_inputs):
            self.weights[i] += alpha * (output - self.output) * self.inputs[i]
    # adjust the margin error until to get the values function
    def training(self, alpha, times, data):
        self.grad_weigths = [self.weights]
        for j in range(times):
            for i in range(0,len(data[0])):
                self.propagation( data[ i, 0:len(data)-1 ] )
                self.update( alpha, data[ i, len(data)-1 ] )
                self.grad_weigths = np.concatenate( (self.grad_weigths, [self.weights] ), axis=0 )
    # gui representation
    def data_graphics(self):
        plb.plot(self.grad_weigths[:,0], 'k')
        plb.plot(self.grad_weigths[:,1], 'g')
        plb.plot(self.grad_weigths[:,2], 'r')
        plb.show()
        
""" Main """
learn_factor = 0.5
n_inputs = 3
training_times = 1000
perceptron_and = Perceptron(n_inputs)
    # AND table 
matrix_values = np.array( [ [0, 0, 1, 0], \
                            [0, 1, 1, 0], \
                            [1, 0, 1, 0], \
                            [1, 1, 1, 1] ] )
perceptron_and.training(learn_factor, training_times, matrix_values)
perceptron_and.data_graphics()