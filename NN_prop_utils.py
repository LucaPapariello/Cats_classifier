import numpy as np
import matplotlib.pyplot as plt
import h5py


# --------------------------------------------------------------------------------------------------
# First we need a function to load the data into the notebook.

def load_data():
    # Load the training set.
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])   # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])   # your train set labels
    
    # Load the test set.
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])      # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])      # your test set labels
    
    classes = np.array(test_dataset["list_classes"][:])            # the list of classes
    
    # Reshape the training and test sets
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# --------------------------------------------------------------------------------------------------
# Second, we define the activation functions (sigmoid and relu) needed for the forward propagation
# and their derivatives for the backpropagation.

def sigmoid(Z):
    """
    Implements the sigmoid activation function in numpy.
    
    Arguments:
    Z -- Output of the linear layer, numpy array of any shape.
    
    Returns:
    A -- output of sigmoid(z), same shape as Z.
    cache -- returns Z as well, useful during backpropagation.
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache



def relu(Z):
    """
    Implements the RELU activation function in numpy.

    Arguments:
    Z -- Output of the linear layer, numpy array of any shape.

    Returns:
    A -- Post-activation parameter, same shape as Z.
    cache -- returns Z as well, useful during backpropagation.
    """
    
    A = np.maximum(0,Z)
    
    cache = Z 
    return A, cache



def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape.
    cache -- 'Z' where we store for computing backward propagation efficiently.

    Returns:
    dZ -- Gradient of the cost with respect to Z.
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))      # The sigmoid function.
    dZ = dA * s * (1-s)       # Its derivative is simply s*(1-s).
    
    return dZ



def relu_backward(dA, cache):
    """
    Implements the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape.
    cache -- 'Z' where we store for computing backward propagation efficiently.

    Returns:
    dZ -- Gradient of the cost with respect to Z.
    """
    
    Z = cache
    dZ = np.array(dA, copy=True)     # Just converting dZ to the correct object.
    
    # When z <= 0, then dZ is 0.
    dZ[Z <= 0] = 0
    
    return dZ


# --------------------------------------------------------------------------------------------------
# Third, we provide a function that initializes the parameters (weights and biases) to
# random numbers. The weigths W_j's are also multiplied with a small number.

def initialize_parameters(layer_dims):
    """
    Initializes the parameters of the neural network to random values using the
    `Xavier initialization` (see: X. Glorot and Y. Bengio, 2010).
        
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network.
                      E.g. layers_dims = [n_x, n_h1, n_h2, n_h3, n_y] gives a 4-layer model
        
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1]).
    bl -- bias vector of shape (layer_dims[l], 1).
    """
    
    parameters = {}                # Initialize a dictionary to store the parameters.
    L = len(layer_dims)            # Number of layers in the network.
    
    for l in range(1, L):
        # The weigths are scaled by a factor sqrt(1/layer_dims[l-1]), i.e. by the
        # sqrt of the previous layer size.
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters


# --------------------------------------------------------------------------------------------------
# Fourth, we provide the functions for the forward propagation.

def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation, i.e. Z = W*A + b.
        
    Arguments:
    A -- activations from previous layer, of shape (size of previous layer, number of examples).
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer).
    b -- bias vector, numpy array of shape (size of the current layer, 1).
        
    Returns:
    Z -- the input of the activation function (i.e. pre-activation parameter).
    cache -- a python dictionary containing "A", "W" and "b", stored for computing the backward pass.
    """
    
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache



def linear_activation_forward(A_prev, W, b, activation):
    """
    Implements the forward propagation for the LINEAR->ACTIVATION layer, i.e. A = g(Z), with
    g the activation function.
        
    Arguments:
    A_prev -- activations from previous layer, of shape (size of previous layer, number of examples).
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer).
    b -- bias vector, numpy array of shape (size of the current layer, 1).
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu".
        
    Returns:
    A -- the output of the activation function (i.e. post-activation value).
    cache -- a python dictionary containing "linear_cache" and "activation_cache", stored for
    computing the backward pass.
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)      # linear_cache stores "A_prev, W, b".
        A, activation_cache = sigmoid(Z)                    # activation_cache stores Z.
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)      # linear_cache stores "A_prev, W, b".
        A, activation_cache = relu(Z)                       # activation_cache stores Z.
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache



def L_model_forward(X, parameters):
    """
    Implements the full forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
        
    Arguments:
    X -- data, numpy array of shape (input size, number of examples).
    parameters -- output of initialize_parameters(), i.e. a python dictionary.
        
    Returns:
    AL -- last post-activation value.
    caches -- list of caches containing every cache of linear_activation_forward(); there are L-1
              of them, indexed from 0 to L-1.
    """
    
    caches = []
    A = X                               # The first value for A comes from the input layer, i.e. X.
    L = len(parameters) // 2            # Number of layers in the neural network.
                                        # // is for integer division
    
    # Implement [LINEAR -> RELU]*(L-1) for all the layers except the last one.
    # Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)],
                                             parameters['b' + str(l)],
                                             activation='relu')
        caches.append(cache)


    # Implement LINEAR -> SIGMOID for the last layer.
    # Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A,
                                          parameters['W' + str(L)],
                                          parameters['b' + str(L)],
                                          activation='sigmoid')
    caches.append(cache)
        
    assert(AL.shape == (1,X.shape[1]))
                                          
    return AL, caches


# --------------------------------------------------------------------------------------------------
# Here we compute the cross-entropy cost function.

def compute_cost(AL, Y):
    """
    Implements the cross-entropy cost function.
        
    Arguments:
    AL -- probability vector corresponding to our label predictions, of shape (1, number of examples).
    Y -- true "label" vector (1 if cat, 0 if non-cat), of shape (1, number of examples).
        
    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]      # Number of examples
    
    # Compute loss from AL and Y.
    cost = -np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)
    cost /= m
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost


# --------------------------------------------------------------------------------------------------
#  We now provide the functions for the backpropagation.

def linear_backward(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer (layer l).
        
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape
               as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    # Here we code the formulas for dW = dL/dW (layer l), db = dL/db (layer l) and dA = dL/dA (layer l-1).
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db



def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.
        
    Arguments:
    dA -- post-activation gradient for current layer l.
    cache -- tuple of values (linear_cache, activation_cache) we stored for computing backward propagation.
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu".
        
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape
               as A_prev.
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W.
    db -- Gradient of the cost with respect to b (current layer l), same shape as b.
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        # Remember that: dZ = dA*g'(Z).
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    elif activation == "sigmoid":
        # Remember that: dZ = dA*g'(Z).
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db



def L_model_backward(AL, Y, caches):
    """
    Implements the backward propagation for the [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID computation.
        
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward()).
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat).
    caches -- list of caches containing:
    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2);
    the cache of linear_activation_forward() with "sigmoid" (there is one, with index L-1).
        
    Returns:
    grads -- A dictionary with the gradients:
    grads["dA" + str(l)] = ...
    grads["dW" + str(l)] = ...
    grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)    # The number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)   # After this line, Y has the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # Derivative of the cost with respect to AL
    
    # Lth layer (SIGMOID -> LINEAR) gradients.
    # Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"].
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] =\
            linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)],
                                                                    current_cache,
                                                                    activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads


# --------------------------------------------------------------------------------------------------
#  We update the parameters of the model using gradient descent.

def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent.
        
    Arguments:
    parameters -- python dictionary containing your parameters.
    grads -- python dictionary containing our gradients, output of L_model_backward.
        
    Returns:
    parameters -- python dictionary containing the updated parameters:
    parameters["W" + str(l)] = ...
    parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2    # Number of layers in the neural network
    
    # Update rule for each parameter.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters


# --------------------------------------------------------------------------------------------------
#  We update the parameters of the model using gradient descent.

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
        
    Arguments:
    X -- data set of examples you would like to label.
    parameters -- parameters of the trained model.
        
    Returns:
    p -- predictions for the given dataset X.
    """
    
    m = X.shape[1]                # Number of examples
    n = len(parameters) // 2      # Number of layers in the neural network
    p = np.zeros((1,m))           # Prediction vector
    
    # Forward propagation
    probs, caches = L_model_forward(X, parameters)
    
    
    # Convert the prediction probabilities "probs" to 0/1 predictions
    for i in range(0, probs.shape[1]):
        # If probs is larger than 0.5, than we set the prediction to 1;
        # if it's smaller to 0.
        if probs[0][i] > 0.5:
            p[0][i] = 1
        else:
            p[0][i] = 0
    
    # Print results
    #print("probas: " + str(probas))
    #print("predictions: " + str(p))
    #print("true labels: " + str(y))
    #print("Accuracy: "  + str(np.sum((p == y)/m)))
    print("Accuracy: " + str(100 - np.mean(np.abs(p - y)) * 100))
    return p
