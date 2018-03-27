from typing import Dict

import numpy as np

from utils import prepare_mnist

def init_params(layer_dims):
    params = {}
    params['W'] = [np.random.randn(l, k) for l, k in zip(layer_dims[1:], layer_dims)]
    params['b'] = [np.random.randn(l, 1) for l in layer_dims[1:]]
    return params

def forward_prop(X, params):
    assert X.shape[0] == params['W'][0].shape[1]

    cache = {'Z': [], 'A': []}
    A = X
    cache['A'].append(A)
    for W, b in zip(params['W'], params['b']):
        Z = np.dot(W, A) + b
        A = sigmoid(Z)
        cache['Z'].append(Z)
        cache['A'].append(A)

    Y_pred = A

    return Y_pred, cache

def back_prop(Y, Y_pred, params, cache):
    L = len(params['W'])
    n_samples = Y.shape[1]

    grads = {'dW': [], 'db': []}
    dZ = mse_cost_deriv(Y, Y_pred) * sigmoid_deriv(cache['Z'][-1])
    dW = np.dot(dZ, cache['A'][-2].T)
    db = np.sum(dZ, axis=1, keepdims=True)
    grads['dW'].append(dW)
    grads['db'].append(db)

    for l in range(2, L + 1):
        dZ = np.dot(params['W'][-l+1].T, dZ) * sigmoid_deriv(cache['Z'][-l])
        dW = np.dot(dZ, cache['A'][-l-1].T)
        db = np.sum(dZ, axis=1, keepdims=True)

        grads['dW'].append(dW)
        grads['db'].append(db)

    grads['dW'] = grads['dW'][::-1]
    grads['db'] = grads['db'][::-1]

    return grads

def update_params(params, grads, learning_rate, mini_batch_size):
    updated = {}
    updated['W'] = [W - learning_rate / mini_batch_size * dW
                    for W, dW in zip(params['W'], grads['dW'])]
    updated['b'] = [b - learning_rate / mini_batch_size * db
                    for b, db in zip(params['b'], grads['db'])]
    return updated

def gradient_descent(X_train, Y_train, X_test, Y_test, params, n_epochs, mini_batch_size, learning_rate):
    n_train = X_train.shape[1]
    n_test = X_test.shape[1]

    for i in range(n_epochs):
        shuffle_idx = np.random.permutation(n_train)
        X_shuffled = X_train[:, shuffle_idx]
        Y_shuffled = Y_train[:, shuffle_idx]

        for start in range(0, n_train, mini_batch_size):
            end = start + mini_batch_size
            X_minibatch = X_shuffled[:, start:end]
            Y_minibatch = Y_shuffled[:, start:end]

            Y_pred, cache = forward_prop(X_minibatch, params)

            grads = back_prop(Y_minibatch, Y_pred, params, cache)

            params = update_params(params, grads, learning_rate, mini_batch_size)

        Y_pred, cache = forward_prop(X_test, params)
        Y_pred = np.argmax(Y_pred, axis=0)
        n_errors = np.sum(Y_test == Y_pred)
        print(f'Epoch {i}: {n_errors}/{n_test}')

    return params

def model(X_train, Y_train, X_test, Y_test, layer_dims, n_epochs, mini_batch_size, learning_rate):
    params = init_params(layer_dims)

    params = gradient_descent(X_train, Y_train, X_test, Y_test, params, n_epochs, mini_batch_size, learning_rate)

    return params

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def mse_cost(Y, Y_pred):
    return 1.0 / Y.shape[1] * np.sum(np.linalg.norm(Y - Y_pred, axis=0))

def mse_cost_deriv(Y, Y_pred):
    return (Y_pred - Y)


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = prepare_mnist(n_train=60000)

    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]

    model(X_train, Y_train, X_test, Y_test, [n_x, 10, 10, n_y], 30, 10, 3.0)

