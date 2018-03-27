import numpy as np

from nnet import init_params, forward_prop, sigmoid, back_prop, mse_cost_deriv, sigmoid_deriv, update_params

def test_init_params():
    layer_dims = [3, 20, 5, 4]
    params = init_params(layer_dims)

    Ws = params['W']
    bs = params['b']

    n_layers = len(layer_dims) - 1

    assert len(Ws) == len(bs)
    assert len(Ws) == n_layers
    assert all(bs[l].shape[1] == 1 for l in range(n_layers))
    assert all(Ws[l].shape[0] == bs[l].shape[0] for l in range(n_layers))
    assert all(Ws[l].shape[0] == Ws[l + 1].shape[1] for l in range(n_layers - 1))

def test_forward_prop():
    np.random.seed(1)

    X = np.random.randn(3, 3)

    params = init_params([3, 2, 1])

    W1, W2 = params['W']
    b1, b2 = params['b']

    Y_pred, cache = forward_prop(X, params)

    A0 = X
    Z1 = np.dot(W1, A0) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (Z1 == cache['Z'][0]).all()
    assert (A1 == cache['A'][1]).all()
    assert (Z2 == cache['Z'][1]).all()
    assert (A2 == Y_pred).all()

def test_back_prop():
    np.random.seed(1)

    X = np.random.randn(3, 3)
    Y = np.array([1, 0, 1])

    params = init_params([3, 2, 1])

    Y_pred, cache = forward_prop(X, params)

    grads = back_prop(Y, Y_pred, params, cache)

    W1, W2 = params['W']
    b1, b2 = params['b']
    A0, A1, A2 = cache['A']
    Z1, Z2 = cache['Z']

    dZ2 = mse_cost_deriv(Y, Y_pred) * sigmoid_deriv(Z2)
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * sigmoid_deriv(Z1)
    dW1 = np.dot(dZ1, A0.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    assert (dW1 == grads['dW'][0]).all()
    assert (dW2 == grads['dW'][1]).all()
    assert (db1 == grads['db'][0]).all()
    assert (db2 == grads['db'][1]).all()

def test_update_params():
    np.random.seed(1)

    X = np.random.randn(3, 3)
    Y = np.array([1, 0, 1])
    learning_rate = 0.1

    params = init_params([3, 2, 1])
    Y_pred, cache = forward_prop(X, params)
    grads = back_prop(Y, Y_pred, params, cache)
    updated = update_params(params, grads, learning_rate, 3)

    W1, W2 = params['W']
    b1, b2 = params['b']
    dW1, dW2 = grads['dW']
    db1, db2 = grads['db']

    W1 -= learning_rate / 3 * dW1
    W2 -= learning_rate / 3 * dW2
    b1 -= learning_rate / 3 * db1
    b2 -= learning_rate / 3 * db2

    assert (W1 == updated['W'][0]).all()
    assert (W2 == updated['W'][1]).all()
    assert (b1 == updated['b'][0]).all()
    assert (b2 == updated['b'][1]).all()

if __name__ == '__main__':
    test_init_params()
    test_forward_prop()
    test_back_prop()
    test_update_params()

