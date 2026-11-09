from itertools import product
import numpy as np
from xor_demo import predict, train


def test_xor_truth_table_and_reproducibility():
    X = np.array(list(product([0, 1], repeat=4)), dtype=float)
    y = (X[:, 1].astype(int) ^ X[:, 3].astype(int)).reshape(-1, 1)
    first = train(X, y, iterations=4000)
    second = train(X, y, iterations=4000)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.array_equal(predict(X, *first) >= .5, y)
