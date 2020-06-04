import h5py
import numpy as np
import tensorflow as tf


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_x = np.array(train_dataset["train_set_x"][:])
    train_y = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_x = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:])

    # 归一化数据集
    train_x = train_x / 255
    test_x = test_x / 255
    
    return train_x, train_y, test_x, test_y


def cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu, input_shape=[64, 64, 3]),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', ),
        tf.keras.layers.Conv2D(128, [3, 3], padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', ),
        tf.keras.layers.Conv2D(256, [3, 3], padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same', ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(24, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.8),
        tf.keras.layers.Dense(6, activation=tf.nn.softmax)])

    return model


def random_mini_batch(X, Y, mini_batch_size):
    np.random.seed(0)
    m = X.shape[0]
    num_complete_mini_batch = m // mini_batch_size
    
    permutation = list(np.random.permutation(m))
    X_shuffle = X[permutation, :, :, :]
    Y_shuffle = Y[permutation]
    
    mini_batches = []
    
    for i in range(num_complete_mini_batch):
        X_mini_batch = X_shuffle[i * mini_batch_size:(i + 1) * mini_batch_size, :, :, :]
        Y_mini_batch = Y_shuffle[i * mini_batch_size:(i + 1) * mini_batch_size]
        mini_batches.append((X_mini_batch, Y_mini_batch))
    
    if m % mini_batch_size:
        X_mini_batch = X_shuffle[num_complete_mini_batch * mini_batch_size:, :, :, :]
        Y_mini_batch = Y_shuffle[num_complete_mini_batch * mini_batch_size:]
        mini_batches.append((X_mini_batch, Y_mini_batch))
    
    return mini_batches


def loss(Y, pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(Y, pred)
    return tf.reduce_mean(loss)


def accuracy(Y, pred):
    prediction = tf.argmax(pred, axis=1)
    return np.sum(prediction == Y)