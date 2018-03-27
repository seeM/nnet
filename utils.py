import numpy as np

from mnist import download_mnist, read_mnist

def image_to_vec(images):
    """Flatten image datasets from 3D to 2D."""
    return images.reshape((images.shape[0], images.shape[1] * images.shape[2]))


def one_hot_encode(labels, n_classes=10):
    n_samples = labels.shape[1]
    encoded = np.zeros((n_classes, n_samples))
    encoded[labels, np.arange(n_samples)] = 1
    return encoded

def prepare_mnist(n_train):
    download_mnist()
 
    X, Y = read_mnist() 
    X = image_to_vec(X) 
    X = X / 255 
    X = X.T 
    Y = Y.T 
 
    n_samples = X.shape[1] 
    shuffle_idx = np.random.permutation(n_samples) 
    X_shuffled = X[:, shuffle_idx] 
    Y_shuffled = Y[:, shuffle_idx] 
 
    X_train = X_shuffled[:, :n_train] 
    Y_train = Y_shuffled[:, :n_train] 
    X_test = X_shuffled[:, n_train:] 
    Y_test = Y_shuffled[:, n_train:] 
 
    Y_train = one_hot_encode(Y_train) 

    return X_train, Y_train, X_test, Y_test

