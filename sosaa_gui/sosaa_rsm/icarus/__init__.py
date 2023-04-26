from __future__ import annotations

import abc
from collections import namedtuple

IcarusPrediction = namedtuple(
    "IcarusPrediction",
    [
        "prediction",
        "uncertainty",
        "confidence",
    ],
)


class IcarusRSM(abc.ABC):
    @abc.abstractmethod
    def fit(
        self,
        X_train,  #: np.ndarray,
        Y_train,  #: np.ndarray,
        X_valid,  #: np.ndarray,
        Y_valid,  #: np.ndarray,
        rng,  #: np.random.Generator,
        **kwargs,
    ) -> IcarusRSM:
        return self

    @abc.abstractmethod
    def predict(
        self,
        X_test,  #: np.ndarray,
        rng,  #: np.random.Generator,
        **kwargs,
    ) -> IcarusPrediction:
        return None
