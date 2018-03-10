from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        F = num_filters
        C_conv = input_dim[0]
        HH, WW = filter_size, filter_size
        H, W = input_dim[1], input_dim[2]
        
        #D_fc = input_dim[0] * input_dim[1] * input_dim[2]
        D = int(F * H * W / 4) # assume Conv layer keeps the dim 
        H_fc = hidden_dim
        C_fc = num_classes
        self.params['W1'] = np.random.normal(0.0, weight_scale, [F, C_conv, HH, WW])
        self.params['W2'] = np.random.normal(0.0, weight_scale, [D, H_fc])
        self.params['W3'] = np.random.normal(0.0, weight_scale, [H_fc, C_fc])

        self.params['b1'] = np.zeros(F)
        self.params['b2'] = np.zeros(H_fc)
        self.params['b3'] = np.zeros(C_fc)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        
        # conv - relu - 2x2 max pool - (affine - relu) - affine - softmax
        #out_conv, cache_conv, = conv_forward_naive(X, W1, b1, conv_param)
        out_conv, cache_conv, = conv_forward_fast(X, W1, b1, conv_param)
        out_relu, cache_relu = relu_forward(out_conv)
        #out_max, cache_max = max_pool_forward_naive(out_relu, pool_param)
        out_max, cache_max = max_pool_forward_fast(out_relu, pool_param)
        out_aff, cache_aff = affine_relu_forward(out_max, W2, b2)
        out_cnn, cache_cnn = affine_forward(out_aff, W3, b3)
        scores = out_cnn


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, grad_cnn = softmax_loss(out_cnn, y)
        loss += self.reg * np.sum(W3 * W3)
        
        grad_aff, grads['W3'], grads['b3'] = affine_backward(grad_cnn, cache_cnn)
        grad_max, grads['W2'], grads['b2'] \
                                = affine_relu_backward(grad_aff, cache_aff)
        #grad_relu = max_pool_backward_naive(grad_max, cache_max)
        grad_relu = max_pool_backward_fast(grad_max, cache_max)
        grad_conv = relu_backward(grad_relu, cache_relu)
        #_, grads['W1'], grads['b1'] = conv_backward_naive(grad_conv, cache_conv)
        _, grads['W1'], grads['b1'] = conv_backward_fast(grad_conv, cache_conv)


        ###########################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
