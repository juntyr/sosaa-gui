import datetime
from collections import namedtuple
from pathlib import Path

from ..dataset import load_and_cache_dataset
from ..icarus import IcarusRSM

TrainTestEvaluation = namedtuple(
    "TrainTestEvaluation",
    [
        "train_mse",
        "train_mae",
        "train_r2",
        "train_rmsce",
        "test_mse",
        "test_mae",
        "test_r2",
        "test_rmsce",
    ],
)


def train_and_cache_model(
    dt: datetime.datetime,
    clump: float,
    datasets: dict,
    models: dict,
    cls,
    rng,  #: np.random.RandomState,
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    overwrite_model: bool,
    progress=None,
    **kwargs,
) -> IcarusRSM:
    import joblib

    if isinstance(dt, tuple) or isinstance(dt, list):
        dt = tuple(sorted(dt))

    model_key = (cls.__name__, dt, clump)

    cached = models.get(model_key)

    if cached is not None:
        return cached

    if Path(model_path).exists() and not overwrite_model:
        model = joblib.load(model_path)

        models[model_key] = model

        return model

    dataset = load_and_cache_dataset(
        dt, clump, datasets, input_dir, output_dir, progress=progress
    )

    model = cls().fit(
        X_train=dataset.X_train,
        Y_train=dataset.Y_train,
        X_valid=dataset.X_valid,
        Y_valid=dataset.Y_valid,
        rng=rng,
        progress=progress,
        **kwargs,
    )

    joblib.dump(model, model_path)

    models[model_key] = model

    return model
