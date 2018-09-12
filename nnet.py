# -*- coding: utf-8 -*-
"""
Neural Net for Expert Behavior Cloning
"""

import numpy as np

def AffineForward(A, W, b):
    """
    Computes affine transformation Z = AW + b.
    
    Args:
        A (n x d array): Batch of data
        W (d x d' array): Layer Weight
        b (size d' array): Bias
    
    Returns:
        n x d' array: Z
        tuple: Current (A, W, b)
    """
    return A.dot(W) + b, (A, W, b)

def AffineBackward(dZ, cache):
    """
    Computes the gradients of loss L with respect to forward inputs A, W, b.
    
    Args:
        dZ: Gradient
        cache: A, W, b
    returns
        dA, dW, db
    """
    A, W, b = cache
    dA = dZ.dot(W.T)
    dW = A.T.dot(dZ)
    db = dZ.sum(axis=0)
    return dA, dW, db    

def ReLUForward(Z):
    """
    Computes Elementwise ReLU of Z
    
    Args:
        Z (n x d' array): A Batch

    Returns:
        n x d' array: ReLU output
        cache: value of Z
    """
    A = Z * (Z>=0)
    return A, Z

def ReLUBackward(dA, cache):
    """
    Computes gradient of Z with respect to loss

    Args:
        Gradient dZ
        cache (n x d' array): Cache object from forward

    Returns:
          gradient of Z with respect to loss L
    """
    return dA * (cache>=0)

def CrossEntropy(F, y):
    """
    Computes the loss function L and the gradients of the loss with
    respect to the scores F

    Args:
        F: logits
        y: Target classes

    Returns:
        loss L
        gradients dlogits
    """
    n = y.shape[0]
    Fy = np.zeros(n)
    jey = np.zeros(F.shape)
    for i in range(n):
        Fy[i] = F[i, int(y[i, 0])]
        jey[i, int(y[i, 0])] = 1
    L = -1/n*(Fy - np.log(np.exp(F).sum(axis=1))).sum()
    
    dF = -1/n*(jey - np.divide(np.exp(F), np.exp(F).sum(axis=1).reshape((F.shape[0], 1))))
    return np.array([[L]]), dF

def ThreeNetwork(X, W, B, y, test):
    """
    Three Layer Network
    """
    W1, W2, W3 = W
    B1, B2, B3 = B
    
    Z1, acache1 = AffineForward(X, W1, B1)
    A1, rcache1 = ReLUForward(Z1)
    Z2, acache2 = AffineForward(A1, W2, B2)
    A2, rcache2 = ReLUForward(Z2)
    F, acache3 = AffineForward(A2, W3, B3)
    if test == True:
        classifications = np.argmax(F, axis=1)
        return classifications
    loss, dF = CrossEntropy(F, y)
    dA2, dW3, db3 = AffineBackward(dF, acache3)
    dZ2 = ReLUBackward(dA2, rcache2)
    dA1, dW2, db2 = AffineBackward(dZ2, acache2)
    dZ1 = ReLUBackward(dA1, rcache1)
    dX, dW1, db1 = AffineBackward(dZ1, acache1)
    #use gradient descent to update parameters
    #using arbitrary learning rate
    n = 0.1
    W1 = W1-n*dW1
    W2 = W2-n*dW2
    W3 = W3-n*dW3
    B1 = B1-n*db1
    B2 = B2-n*db2
    B3 = B3-n*db3
    
    W = (W1, W2, W3)
    B = (B1, B2, B3)
    
    return loss, W, B

def FourNetwork(X, W, B, y, test):
    """
    A four layer neural network with 256 units per layer (except the last layer, which has 3) with a learning rate of 0.1.
    
    Args:
        X (array): Data Features
        W (tuple): Four weight arrays
        B (tuple): Four biases
        y (Training): Data classes
        test (bool): Whether training or testing
        
    Returns
        if test:
            array: estimated classifications
        else:
            float: loss
            tuple: weight arrays
            tuple: biases
    """
    W1, W2, W3, W4 = W
    B1, B2, B3, B4 = B
    Z1, acache1 = AffineForward(X, W1, B1)
    A1, rcache1 = ReLUForward(Z1)
    Z2, acache2 = AffineForward(A1, W2, B2)
    A2, rcache2 = ReLUForward(Z2)
    Z3, acache3 = AffineForward(A2, W3, B3)
    A3, rcache3 = ReLUForward(Z3)
    F, acache4 = AffineForward(A3, W4, B4)
    if test == True:
        classifications = np.argmax(F, axis=1)
        return classifications
    loss, dF = CrossEntropy(F, y)
    dA3, dW4, db4 = AffineBackward(dF, acache4)
    dZ3 = ReLUBackward(dA3, rcache3)
    dA2, dW3, db3 = AffineBackward(dZ3, acache3)
    dZ2 = ReLUBackward(dA2, rcache2)
    dA1, dW2, db2 = AffineBackward(dZ2, acache2)
    dZ1 = ReLUBackward(dA1, rcache1)
    dX, dW1, db1 = AffineBackward(dZ1, acache1)
    #use gradient descent to update parameters
    #using arbitrary learning rate
    n = 0.1
    W1 = W1-n*dW1
    W2 = W2-n*dW2
    W3 = W3-n*dW3
    W4 = W4-n*dW4
    B1 = B1-n*db1
    B2 = B2-n*db2
    B3 = B3-n*db3
    B4 = B4-n*db4

    W = (W1, W2, W3, W4)
    B = (B1, B2, B3, B4)
    
    return loss, W, B

def MinibatchGDThree(data, epoch, batch_size, weight_scale):
    n = batch_size
    xcols = data.shape[1]-1
    wcols = 3
    
    w1shape = (xcols, 3)
    w2shape = (3, 3)
    w3shape = (3, 3)
    bshape = (1, wcols)
    
    W1 = np.random.rand(w1shape[0]*w1shape[1]).reshape(w1shape) * weight_scale
    W2 = np.random.rand(w2shape[0]*w2shape[1]).reshape(w2shape) * weight_scale
    W3 = np.random.rand(w3shape[0]*w3shape[1]).reshape(w3shape) * weight_scale
    
    W = (W1, W2, W3)

    B1 = np.zeros(bshape)
    B2 = np.zeros(bshape)
    B3 = np.zeros(bshape)
    
    B = (B1, B2, B3)

    N = data.shape[0]

    records = []
    for e in range(epoch):
        if e%50 == 0:
            print("At epoch {}".format(e))
        #shuffle data
        datacopy = data.copy()
        np.random.shuffle(datacopy)
        for i in range(int(N/n)):
          X = datacopy[(i)*n:(i+1)*n, :-1]
          y = datacopy[(i)*n:(i+1)*n, -1:]
          loss,W, B = ThreeNetwork(X, W, B, y, False)
          records.append([loss, W, B])

    return np.array(records)

def MinibatchGDFour(data, epoch, batch_size, weight_scale):
    n = batch_size
    xcols = data.shape[1]-1
    units1 = 64
    units2 = 64
    units3 = 64
    units4 = 3
    
    w1shape = (xcols, units1)
    w2shape = (units1, units2)
    w3shape = (units2, units3)
    w4shape = (units3, units4)
    
    b1shape = (1, units1)
    b2shape = (1, units2)
    b3shape = (1, units3)
    b4shape = (1, units4)

    # W1 = np.random.rand(w1shape[0]*w1shape[1]).reshape(w1shape) * weight_scale
    # W2 = np.random.rand(w2shape[0]*w2shape[1]).reshape(w2shape) * weight_scale
    # W3 = np.random.rand(w3shape[0]*w3shape[1]).reshape(w3shape) * weight_scale
    # W4 = np.random.rand(w4shape[0]*w4shape[1]).reshape(w4shape) * weight_scale

    W1 = np.random.uniform(-0.2, 0.2, w1shape)
    W2 = np.random.uniform(-0.2, 0.2, w2shape)
    W3 = np.random.uniform(-0.2, 0.2, w3shape)
    W4 = np.random.uniform(-0.2, 0.2, w4shape)

    W = (W1, W2, W3, W4)
    
    B1 = np.zeros(b1shape)
    B2 = np.zeros(b2shape)
    B3 = np.zeros(b3shape)
    B4 = np.zeros(b4shape)
    
    B = (B1, B2, B3, B4)
    
    N = data.shape[0]

    records = []
    for e in range(epoch):
        if e%50 == 0:
            print("At epoch {}".format(e))
        #shuffle data
        datacopy = data.copy()
        np.random.shuffle(datacopy)
        for i in range(int(N/n)):
            X = datacopy[(i)*n:(i+1)*n, :-1]
            y = datacopy[(i)*n:(i+1)*n, -1:]
            loss, W, B = FourNetwork(X, W, B, y, False)
        records.append([loss, W, B])

    return np.array(records)
    