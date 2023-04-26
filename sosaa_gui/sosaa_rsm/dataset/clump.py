"""
Clumped 0/1 sampler using a Markov Process

P(0) = p and P(1) = 1-p
clump = 0 => IID samples
clump -> 1 => highly correlated samples

"""


class Clump:
    def __init__(self, p=0.5, clump=0.0, rng=None):
        import numpy as np

        a = 1 - (1 - p) * (1 - clump)
        b = (1 - a) * p / (1 - p)

        self.C = np.array([[a, 1 - a], [b, 1 - b]])

        self.i = 0 if rng.random() < p else 1

    def sample(self, rng):
        p = self.C[self.i, 0]
        u = rng.random()

        self.i = 0 if u < p else 1

        return self.i

    def steady(self, X):
        import numpy as np

        return np.matmul(X, self.C)


def train_test_split(
    X, Y, test_size=0.25, random_state=None, shuffle=True, clump=0.0
):
    import numpy as np
    import pandas as pd

    assert len(X) == len(Y)
    assert type(X) == type(Y)
    assert test_size > 0.0
    assert test_size < 1.0
    assert random_state is not None
    assert clump >= 0.0
    assert clump < 1.0

    c = Clump(p=test_size, clump=clump, rng=random_state)

    if isinstance(X, pd.DataFrame):
        assert X.index.values.shape == Y.index.values.shape

        # Split only based on the first-level index instead of flattening
        n1 = len(X.index.levels[1])
        n0 = len(X) // n1

        C = np.array([c.sample(random_state) for _ in range(n0)])
        (I_train,) = np.nonzero(C)
        I_train = np.repeat(I_train, n1) * n1 + np.tile(
            np.arange(n1), len(I_train)
        )
        (I_test,) = np.nonzero(1 - C)
        I_test = np.repeat(I_test, n1) * n1 + np.tile(
            np.arange(n1), len(I_test)
        )
    else:
        C = np.array([c.sample(random_state) for _ in range(len(X))])
        (I_train,) = np.nonzero(C)
        (I_test,) = np.nonzero(1 - C)

    if shuffle:
        random_state.shuffle(I_train)
        random_state.shuffle(I_test)

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[I_train]
        X_test = X.iloc[I_test]

        Y_train = Y.iloc[I_train]
        Y_test = Y.iloc[I_test]
    else:
        X_train = X[I_train]
        X_test = X[I_test]

        Y_train = Y[I_train]
        Y_test = Y[I_test]

    return X_train, X_test, Y_train, Y_test
