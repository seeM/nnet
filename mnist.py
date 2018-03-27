import os
import struct
import urllib.request
import gzip

import numpy as np

BASE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES_URL = BASE_URL + 'train-images-idx3-ubyte.gz'
TRAIN_LABELS_URL = BASE_URL + 'train-labels-idx1-ubyte.gz'
TEST_IMAGES_URL = BASE_URL + 't10k-images-idx3-ubyte.gz'
TEST_LABELS_URL = BASE_URL + 't10k-labels-idx1-ubyte.gz'

PATH = './data/'
TRAIN_IMAGES_PATH = PATH + 'train-images-idx3-ubyte'
TRAIN_LABELS_PATH = PATH + 'train-labels-idx1-ubyte'
TEST_IMAGES_PATH = PATH + 't10k-images-idx3-ubyte'
TEST_LABELS_PATH = PATH + 't10k-labels-idx1-ubyte'

URLS = [TRAIN_IMAGES_URL, TRAIN_LABELS_URL,
        TEST_IMAGES_URL, TEST_LABELS_URL]

PATHS = [TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH,
         TEST_IMAGES_PATH, TEST_LABELS_PATH]

def download_mnist():
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    for url, path in zip(URLS, PATHS):
        if not os.path.exists(path):
            gz_path = path + '.gz'
            if not os.path.exists(gz_path):
                print('Downloading ' + url)
                with urllib.request.urlopen(url) as response:
                    data = response.read()
                print('Writing to ' + gz_path)
                with open(gz_path, 'wb') as f:
                    f.write(data)
            print('Unzipping to ' + path)
            with gzip.open(gz_path, 'rb') as gz:
                gz_content = gz.read()
            with open(path, 'wb') as f:
                f.write(gz_content)

def read_idx(filename):
    """
    A function that can read MNIST's idx file format into numpy arrays.

    This relies on the fact that the MNIST dataset consistently uses
    unsigned char types with their data segments.

    From: https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def read_mnist():
    images_train = read_idx(TRAIN_IMAGES_PATH)
    images_test = read_idx(TEST_IMAGES_PATH)
    labels_train = read_idx(TRAIN_LABELS_PATH)
    labels_test = read_idx(TEST_LABELS_PATH)

    images = np.concatenate((images_train, images_test), axis=0)
    labels = np.concatenate((labels_train, labels_test), axis=0)
    labels = labels.reshape((-1, 1))

    return images, labels

