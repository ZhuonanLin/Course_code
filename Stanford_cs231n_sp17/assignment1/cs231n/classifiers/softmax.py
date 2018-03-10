import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_d = W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    s = X[i].dot(W)
    s_yi = s[y[i]]
    loss += (-s_yi + np.log(np.sum(np.exp(s))))
    for j in xrange(num_classes):
        flag = np.exp(s[j]) / np.sum(np.exp(s))
        if j == y[i]:
            flag -= 1
        for k in xrange(num_d):
            dW[k, j] += flag * X[i, k]
  loss /= float(num_train)
  loss += reg * np.sum(W * W)
  dW /= float(num_train)
  dW += 2 * reg * W
  ### gradient
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_d = W.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  s = X.dot(W)
  loss = np.sum(-s[np.arange(num_train), y] + np.log(np.sum(np.exp(s), axis=1)))
  loss /= float(num_train)
  #mask = np.zeros(W.shape)
  
  loss += reg * np.sum(W * W)
  mask = np.divide(np.exp(s), np.sum(np.exp(s),axis=1, keepdims=True)).T
  mask[y, np.arange(num_train)] -= 1
  dW = mask.dot(X).T / num_train
  dW += 2 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

