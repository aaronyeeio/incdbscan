import numpy as np
import xxhash
from sklearn.utils.validation import check_array


def hash_(array):
    return xxhash.xxh64_intdigest(array.tobytes()) >> 1


def input_check(X):
    X = np.asarray(X, dtype=float)
    return check_array(X, dtype=float, accept_large_sparse=False) if X.size > 0 else X
