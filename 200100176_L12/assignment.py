import numpy as np
from matplotlib import pyplot as plt
import math
import pickle as pkl

def preprocessing(X):
    """
    Implement Normalization for input image features

    Args:
    X : numpy array of shape (n_samples, n_features)
    
    Returns:
    X_out: numpy array of shape (n_samples, n_features) after normalization
    """
    X_out = None
    
    ## TODO
    X_out = X/np.max(X)
    ## END TODO

    assert X_out.shape == X.shape

    return X_out

def split_data(X, Y, train_ratio=0.8):
    '''
    Split data into train and validation sets
    The first floor(train_ratio*n_sample) samples form the train set
    and the remaining the validation set

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    train_ratio - fraction of samples to be used as training data

    Returns:
    X_train, Y_train, X_val, Y_val
    '''
    # Try Normalization and scaling and store it in X_transformed
    X_transformed = X

    ## TODO
    X_transformed = preprocessing(X)
    ## END TODO

    assert X_transformed.shape == X.shape

    num_samples = len(X)
    indices = np.arange(num_samples)
    num_train_samples = math.floor(num_samples * train_ratio)
    train_indices = np.random.choice(indices, num_train_samples, replace=False)
    val_indices = list(set(indices) - set(train_indices))
    X_train, Y_train, X_val, Y_val = X_transformed[train_indices], Y[train_indices], X_transformed[val_indices], Y[val_indices]
  
    return X_train, Y_train, X_val, Y_val

class FlattenLayer:
    '''
    This class converts a multi-dimensional into 1-d vector
    '''
    def __init__(self, input_shape):
        '''
         Args:
          input_shape : Original shape, tuple of ints
        '''
        self.input_shape = input_shape

    def forward(self, input):
        '''
        Converts a multi-dimensional into 1-d vector
        Args:
          input : training data, numpy array of shape (n_samples , self.input_shape)

        Returns:
          input: training data, numpy array of shape (n_samples , -1)
        '''
        ## TODO
        input = input.reshape(1, 28*28)
        #Modify the return statement to return flattened input
        return input
        ## END TODO
        
    
    def backward(self, output_error, learning_rate):
        '''
        Converts back the passed array to original dimention 
        Args:
        output_error :  numpy array 
        learning_rate: float

        Returns:
        output_error: A reshaped numpy array to allow backward pass
        '''
        ## TODO

        #Modify the return statement to return reshaped array
        return output_error
        ## END TODO
        
        
class FCLayer:
    '''
    Implements a fully connected layer  
    '''
    def __init__(self, input_size, output_size):
        '''
        Args:
         input_size : Input shape, int
         output_size: Output shape, int 
        '''
        self.input_size = input_size
        self.output_size = output_size
        ## TODO
        lower, upper = -(math.sqrt(6.0) / math.sqrt(input_size + output_size)), (math.sqrt(6.0) / math.sqrt(input_size + output_size))
        self.weights = lower + np.random.rand(input_size, output_size)*(upper-lower)
        self.bias = lower + np.random.rand(1, output_size)*(upper-lower)
        ## END TODO

    def forward(self, input):
        '''
        Performs a forward pass of a fully connected network
        Args:
          input : training data, numpy array of shape (n_samples , self.input_size)

        Returns:
           numpy array of shape (n_samples , self.output_size)
        '''
        ## TODO
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
        #Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        # return None
        ## END TODO
        

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network along with updating the parameter 
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        ## TODO
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
        #Modify the return statement to return numpy array resulting from backward pass
        # return None
        ## END TODO
        
        
class ActivationLayer:
    '''
    Implements a Activation layer which applies activation function on the inputs. 
    '''
    def __init__(self, activation, activation_prime):
        '''
          Args:
          activation : Name of the activation function (sigmoid,tanh or relu)
          activation_prime: Name of the corresponding function to be used during backpropagation (sigmoid_prime,tanh_prime or relu_prime)
        '''
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        '''
        Applies the activation function 
        Args:
          input : numpy array on which activation function is to be applied

        Returns:
           numpy array output from the activation function
        '''
        ## TODO
        self.input = input
        self.output = self.activation(self.input)
        return self.output
        #Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        # return None
        ## END TODO
        

    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a fully connected network along with updating the parameter 
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        ## TODO
        #Modify the return statement to return numpy array resulting from backward pass
        return self.activation_prime(self.input) * output_error
        ## END TODO
        
        

class SoftmaxLayer:
    '''
      Implements a Softmax layer which applies softmax function on the inputs. 
    '''
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        '''
        Applies the softmax function 
        Args:
          input : numpy array on which softmax function is to be applied

        Returns:
           numpy array output from the softmax function
        '''
        ## TODO
        self.old = np.exp(input) / np.exp(input).sum(axis=1) [:,None]
        return self.old
        #Modify the return statement to return numpy array of shape (n_samples , self.output_size)
        # return None
        ## END TODO
        
    def backward(self, output_error, learning_rate):
        '''
        Performs a backward pass of a Softmax layer
        Args:
          output_error :  numpy array 
          learning_rate: float

        Returns:
          Numpy array resulting from the backward pass
        '''
        ## TODO
        return self.old * (output_error -(output_error * self.old).sum(axis=1)[:,None])
        #Modify the return statement to return numpy array resulting from backward pass
        # return None
        ## END TODO
        
        
def sigmoid(x):
    '''
    Sigmoid function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying simoid function
    '''
    ## TODO
    x = x.clip(min=-700, max=None)
    A = 1/(1+np.exp(-x))
    #Modify the return statement to return numpy array resulting from backward pass
    return A
    ## END TODO

def sigmoid_prime(x):
    '''
     Implements derivative of Sigmoid function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of Sigmoid function
    '''
    ## TODO
    x = x.clip(min=-700, max=None)
    A = 1/(1+np.exp(-x))
    #Modify the return statement to return numpy array resulting from backward pass
    return A*(1-A)
    ## END TODO

def tanh(x):
    '''
    Tanh function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying tanh function
    '''
    ## TODO
    #Modify the return statement to return numpy array resulting from backward pass
    return np.tanh(x)
    ## END TODO

def tanh_prime(x):
    '''
     Implements derivative of Tanh function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of Tanh function
    '''
    ## TODO
    #Modify the return statement to return numpy array resulting from backward pass
    return 1-np.tanh(x)**2
    ## END TODO

def relu(x):
    '''
    ReLU function 
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying ReLU function
    '''
    ## TODO
    #Modify the return statement to return numpy array resulting from backward pass
    return np.maximum(0,x)
    ## END TODO

def relu_prime(x):
    '''
     Implements derivative of ReLU function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of ReLU function
    '''
    ## TODO
    #Modify the return statement to return numpy array resulting from backward pass
    return np.where(x > 0, 1, 0)
    ## END TODO
    
def cross_entropy(y_true, y_pred):
    '''
    Cross entropy loss 
    Args:
        y_true :  Ground truth labels, numpy array 
        y_true :  Predicted labels, numpy array 
    Returns:
       loss : float
    '''
    ## TODO
    y_pred = y_pred.clip(min=1e-8,max=None)
    return (np.where(y_true==1,-np.log(y_pred), 0)).sum(axis=1)
    #Modify the return statement to return numpy array resulting from backward pass
    # return 0
    ## END TODO

def cross_entropy_prime(y_true, y_pred):
    '''
    Implements derivative of cross entropy function, for the backward pass
    Args:
        x :  numpy array 
    Returns:
        Numpy array after applying derivative of cross entropy function
    '''
    ## TODO
    #Modify the return statement to return numpy array resulting from backward pass
    return np.where(y_true==1,-1/y_pred, 0)
    ## END TODO
    
    
def fit(X_train, Y_train, dataset_name):

    '''
    Create and trains a feedforward network

    Do not forget to save the final model/weights of the feed forward network to a file. Use these weights in the `predict` function 
    Args:
        X_train -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.
        Y_train -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.
        dataset_name -- name of the dataset (flowers or mnist)
    
    '''
     
    #Note that this just a template to help you create your own feed forward network 
    ## TODO

    #define your network
    #This example network below would work only for mnist
    #you will need separtae networks for the two datasets
    network = [
        FCLayer(2048, 100),
        ActivationLayer(sigmoid, sigmoid_prime),
        FCLayer(100, 50),
        ActivationLayer(sigmoid, sigmoid_prime),
        FCLayer(50, 5),
        SoftmaxLayer(5)
    ] # This creates feed forward 


    # Choose appropriate learning rate and no. of epoch
    epochs = 40
    learning_rate = 0.1
    X_train = preprocessing(X_train)

    # Change training loop as you see fit
    ident = np.identity(5)
    for epoch in range(epochs):
        error = 0
        score = 0
        for x, y_true in zip(X_train, Y_train):
            # forward
            x = x.reshape(1,2048)
            output = x
            for layer in network:
                output = layer.forward(output)
            
            # error (display purpose only)
            y_true = ident[y_true,:]
            error += cross_entropy(y_true, output)

            # backward
            output_error = cross_entropy_prime(y_true, output)
            for layer in reversed(network):
                output_error = layer.backward(output_error, learning_rate)

        for x, y_true in zip(X_train, Y_train):
            # forward
            x = x.reshape(1,2048)
            output = x
            for layer in network:
                output = layer.forward(output)

            if y_true == np.argmax(output):
                score += 1
        
        error /= len(X_train)
        score /= len(X_train)
        print(score)
        print('%d/%d, error=%f' % (epoch + 1, epochs, error))

    #Save you model/weights as ./models/{dataset_name}_model.pkl
    filename = "./models/"+dataset_name+"_model.pkl"
    pkl.dump(network, open(filename, 'wb'))
    
    ## END TODO
    
def predict(X_test, dataset_name):
    """

    X_test -- np array of share (num_test, 2048) for flowers and (num_test, 28, 28) for mnist.

    This is the function that we will call from the auto grader. 

    This function should only perform inference, please donot train your models here.

    Steps to be done here:
    1. Load your trained model/weights from ./models/{dataset_name}_model.pkl
    2. Ensure that you read model/weights using only the libraries we have given above.
    3. In case you have saved weights only in your file, itialize your model with your trained weights.
    4. Compute the predicted labels and return it

    Return:
    Y_test - nparray of shape (num_test,)
    """
    Y_test = np.zeros(X_test.shape[0],)

    ## TODO
    i = 0
    filename = "./models/"+dataset_name+"_model.pkl"
    loaded_model = pkl.load(open(filename, 'rb'))

    X_test = preprocessing(X_test)

    for x in X_test:
        # forward
        output = x
        for layer in loaded_model:
            output = layer.forward(output)

        Y_test[i] = np.argmax(output)
        i += 1


    ## END TODO
    assert Y_test.shape == (X_test.shape[0],) and type(Y_test) == type(X_test), "Check what you return"
    return Y_test
    
"""
Loading data and training models
"""
if __name__ == "__main__":    
    dataset = "mnist" 
    with open(f"./data/{dataset}_train.pkl", "rb") as file:
        train_mnist = pkl.load(file)
        print(f"train_x -- {train_mnist[0].shape}; train_y -- {train_mnist[1].shape}")
    
    fit(train_mnist[0],train_mnist[1],'mnist')
    
    dataset = "flowers"
    with open(f"./data/{dataset}_train.pkl", "rb") as file:
        train_flowers = pkl.load(file)
        print(f"train_x -- {train_flowers[0].shape}; train_y -- {train_flowers[1].shape}")
    
    fit(train_flowers[0],train_flowers[1],'flowers')





