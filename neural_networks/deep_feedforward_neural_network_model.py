import numpy as np

class DNN:
    """
    Structure: A Feedforward Neural Network
    Optimization: Gradient Descent Algorithm
    """
    def __init__(self, binary_classification=True) -> None:
        self.cost_during_training = []
        self.binary_classification = binary_classification

    def train(self, X_train, Y_train, layer_dims, epoch=1_000, learning_rate=0.001, minibatch_size=64):
        """
        Train the neural network with given parameters and data X, Y
        layer_dims: list of # of units in hidden layers
        """

        layer_dims.insert(0, X_train.shape[0])
        layer_dims.append((1 if self.binary_classification else Y_train.shape[0]))
        self.layer_dims = layer_dims
        parameters = self.initialize_parameters()
        self.minibatch_size = minibatch_size

        for i in range(epoch):
            for batch_idx in range(0, int(X_train.shape[1] / minibatch_size)):
                # Mini Batch Samples
                if batch_idx == int(X_train.shape[1] / minibatch_size):
                    X = X_train[:, batch_idx*minibatch_size:]
                    Y = Y_train[:, batch_idx*minibatch_size:]
                else:
                    X = X_train[:, batch_idx*minibatch_size: (batch_idx+1)*minibatch_size]
                    Y = Y_train[:, batch_idx*minibatch_size: (batch_idx+1)*minibatch_size]

                # Forward prop and output of the last layer
                forward_vars = self.forward_prop(parameters, X)
                predictions = forward_vars["A"+str(len(layer_dims)-1)]

                # Cost function is applied
                cost_val = self.cost(predictions, Y)
                self.cost_during_training.append(cost_val)
                
                # Backpropagation for calculating gradients and updating parameters
                gradients = self.backward_prop(parameters, forward_vars, Y)
                parameters = self.update_parameters(parameters, gradients, learning_rate)

        self.learned_parameters = parameters

    def predict(self, X):
        """Predict labels for given data X"""
        
        forward_vars = self.forward_prop(self.learned_parameters, X)
        L = len(self.layer_dims) - 1

        return forward_vars["A"+str(L)]

    def initialize_parameters(self):
        """Initialize parameters with He initialization method"""
        
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters["W"+str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2/self.layer_dims[l-1])
            parameters["b"+str(l)] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def forward_prop(self, parameters, X):
        """Propagate forward in network"""
        
        forward_vars = {"A0": X}

        for l in range(1, len(self.layer_dims)):
            forward_vars["Z"+str(l)] = np.dot(parameters["W"+str(l)], forward_vars["A"+str(l-1)]) + parameters["b"+str(l)]
            
            # As Activation Function: Apply relu in hidden layers, and in the output layer if binary then sigmoid else softmax
            if l < len(self.layer_dims) - 1:
                forward_vars["A"+str(l)] = self.relu(forward_vars["Z"+str(l)])
            elif self.binary_classification:
                forward_vars["A"+str(l)] = self.sigmoid(forward_vars["Z"+str(l)])
            else:
                forward_vars["A"+str(l)] = self.softmax(forward_vars["Z"+str(l)])

        return forward_vars

    def backward_prop(self, parameters, forward_vars, Y):
        """Propagate backward in the network with calculating gradients"""
        gradients = {}

        L = len(self.layer_dims) - 1

        for l in range(L, 0, -1):
            m = forward_vars["A"+str(l-1)].shape[1]
            if l == L and self.binary_classification:
                gradients["dA"+str(l)] = -np.divide(Y, forward_vars["A"+str(l)]) + np.divide((1-Y), (1-forward_vars["A"+str(l)]))
                sigmoid_derivative = forward_vars["A"+str(l)] * (1 - forward_vars["A"+str(l)])
                gradients["dZ"+str(l)] = np.multiply(gradients["dA"+str(l)], sigmoid_derivative)
            elif l == L: # Compute derivatives if multi class classification
                gradients["dZ"+str(l)] = forward_vars["A"+str(l)] - Y # derivatives for the last softmax layer
            else:
                relu_derivative = forward_vars["A"+str(l)] > 0
                gradients["dZ"+str(l)] = np.multiply(gradients["dA"+str(l)], relu_derivative)
            
            gradients["dW"+str(l)] = (1/m) * np.dot(gradients["dZ"+str(l)], forward_vars["A"+str(l-1)].T)
            gradients["db"+str(l)] = (1/m) * np.sum(gradients["dZ"+str(l)], axis=1, keepdims=True)
            gradients["dA"+str(l-1)] = np.dot(parameters["W"+str(l)].T, gradients["dZ"+str(l)])

        return gradients

    def cost(self, Y_hat, Y):
        """
        Log Loss is applied
        """
        
        m = Y.shape[1]
        if self.binary_classification: 
            cost_value = (1/m) * np.sum(-(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)))
        else:
            cost_value = (1/m) * np.sum((-1)*Y*np.log(Y_hat))

        return cost_value

    def update_parameters(self, parameters, gradients, learning_rate):
        """Update parameters with gradients"""
        
        for l in range(1, len(self.layer_dims)):
            parameters["W"+str(l)] -= learning_rate * gradients["dW"+str(l)]
            parameters["b"+str(l)] -= learning_rate * gradients["db"+str(l)]

        return parameters

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        T = np.exp(Z)
        T_sum = np.sum(T, axis=0)
        return np.divide(T, T_sum)