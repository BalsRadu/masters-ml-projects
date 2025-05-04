from builtins import object, range

import numpy as np

from ..layer_utils import *
from ..layers import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params[f"W{i + 1}"] = (
                np.random.randn(layer_dims[i], layer_dims[i + 1]) * weight_scale
            )
            self.params[f"b{i + 1}"] = np.zeros(layer_dims[i + 1])
            if self.normalization == "batchnorm" and i < self.num_layers - 1:
                # hidden layer i+1 gets a gamma/beta
                self.params[f"gamma{i + 1}"] = np.ones(layer_dims[i + 1])
                self.params[f"beta{i + 1}"] = np.zeros(layer_dims[i + 1])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Forward pass: affine - relu for L-1 layers
        out = X
        caches = []
        for i in range(self.num_layers - 1):
            W, b = self.params[f"W{i + 1}"], self.params[f"b{i + 1}"]
            # affine -> (batchnorm) -> relu
            if self.normalization == "batchnorm":
                gamma = self.params[f"gamma{i + 1}"]
                beta = self.params[f"beta{i + 1}"]
                bn_param = self.bn_params[i]
                out, cache = affine_bn_relu_forward(out, W, b, gamma, beta, bn_param)
            else:
                out, cache = affine_relu_forward(out, W, b)
            # dropout
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
                # stash the dropout cache alongside the layer cache
                cache = (cache, drop_cache)
            caches.append(cache)

        # last affine
        scores, cache_last = affine_forward(
            out, self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"]
        )
        caches.append(cache_last)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute softmax loss
        N = X.shape[0]
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        # Add regularization
        for i in range(self.num_layers):
            W = self.params[f"W{i + 1}"]
            loss += 0.5 * self.reg * np.sum(W * W)

        # Backward pass
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        grads = {}
        # Last layer gradient
        dout, dW, db = affine_backward(dscores, caches[-1])
        grads[f"W{self.num_layers}"] = (
            dW + self.reg * self.params[f"W{self.num_layers}"]
        )
        grads[f"b{self.num_layers}"] = db

        # hidden layers
        for i in reversed(range(self.num_layers - 1)):
            cache = caches[i]
            # undo dropout first
            if self.use_dropout:
                # unpack our tuple (layer_cache, drop_cache)
                cache, drop_cache = cache
                dout = dropout_backward(dout, drop_cache)
            # then backprop through affine-(bn)-relu
            if self.normalization == "batchnorm":
                dx, dW, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache)
                grads[f"gamma{i + 1}"] = dgamma
                grads[f"beta{i + 1}"] = dbeta
            else:
                dx, dW, db = affine_relu_backward(dout, cache)
            grads[f"W{i + 1}"] = dW + self.reg * self.params[f"W{i + 1}"]
            grads[f"b{i + 1}"] = db
            dout = dx

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    # affine
    a, fc_cache = affine_forward(x, w, b)
    # batchnorm
    an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    # relu
    out, relu_cache = relu_forward(an)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    # relu backward
    dan = relu_backward(dout, relu_cache)
    # batchnorm backward
    da, dgamma, dbeta = batchnorm_backward_alt(dan, bn_cache)
    # affine backward
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
