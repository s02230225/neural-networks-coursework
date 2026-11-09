"""Two-layer NumPy network for XOR of the second and fourth input bits."""
import argparse
from itertools import product

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def add_bias(X):
    return np.column_stack((X, np.ones(len(X))))


def predict(X, w1, w2):
    hidden = sigmoid(add_bias(np.asarray(X, dtype=float)) @ w1)
    return sigmoid(add_bias(hidden) @ w2)


def train(X, y, iterations=4000, seed=1, learning_rate=1.):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) == 0 or y.shape != (len(X), 1):
        raise ValueError('Expected X as a matrix and y as a column vector')
    if iterations < 1 or learning_rate <= 0:
        raise ValueError('Expected positive iteration count and learning rate')
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1, 1, (X.shape[1] + 1, 4))
    w2 = rng.uniform(-1, 1, (5, 1))
    X_bias = add_bias(X)
    for _ in range(iterations):
        hidden = sigmoid(X_bias @ w1)
        hidden_bias = add_bias(hidden)
        output = sigmoid(hidden_bias @ w2)
        delta2 = (y - output) * output * (1 - output)
        delta1 = (delta2 @ w2[:-1].T) * hidden * (1 - hidden)
        w2 += learning_rate * hidden_bias.T @ delta2
        w1 += learning_rate * X_bias.T @ delta1
    return w1, w2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=4000)
    args = parser.parse_args()
    X = np.array(list(product([0, 1], repeat=4)), dtype=float)
    y = (X[:, 1].astype(int) ^ X[:, 3].astype(int)).reshape(-1, 1)
    w1, w2 = train(X, y, args.iterations)
    probabilities = predict(X, w1, w2)
    print('Accuracy on the full XOR truth table:', float(np.mean((probabilities >= .5) == y)))


if __name__ == '__main__':
    main()
