import numpy as np

def matmul(t1, t2):
    return t1.data @ t2.data

def add(t1, t2):
    return t1.data + t2.data

def sub(t1, t2):
    return t1.data - t2.data

def mul(t1, t2):
    return t1.data * t2.data

def div(t1, t2):
    return t1.data / t2.data

def pow(t1, t2):
    return t1.data ** t2.data

def exp(t1):
    return np.exp(t1.data)

def log(t1):
    return np.log(t1.data)

def tanh(t1):
    return np.tanh(t1.data)

def sigmoid(t1):
    return 1 / (1 + np.exp(-t1.data))

def relu(t1):
    return np.maximum(0, t1.data)

def leaky_relu(t1, alpha=0.01):
    return np.maximum(alpha * t1.data, t1.data)

def softmax(t1):
    exps = np.exp(t1.data - np.max(t1.data))
    return exps / np.sum(exps)

def log_softmax(t1):
    exps = np.exp(t1.data - np.max(t1.data))
    return np.log(exps / np.sum(exps))

def cross_entropy(t1, t2):
    return -np.sum(t2.data * np.log(t1.data))

def binary_cross_entropy(t1, t2):
    return -np.sum(t2.data * np.log(t1.data) + (1 - t2.data) * np.log(1 - t1.data))

def mse(t1, t2):
    return np.mean((t1.data - t2.data) ** 2)

def rmse(t1, t2):
    return np.sqrt(np.mean((t1.data - t2.data) ** 2))

def mae(t1, t2):
    return np.mean(np.abs(t1.data - t2.data))

def huber(t1, t2, delta=1.0):
    abs_diff = np.abs(t1.data - t2.data)
    return np.where(abs_diff < delta, 0.5 * abs_diff ** 2, delta * (abs_diff - 0.5 * delta))

def log_likelihood(t1, t2):
    return np.sum(t2.data * np.log(t1.data))

def negative_log_likelihood(t1, t2):
    return -np.sum(t2.data * np.log(t1.data))

def poisson(t1, t2):
    return np.sum(t2.data * np.log(t1.data) - t1.data)