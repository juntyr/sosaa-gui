import datetime
from collections import namedtuple
from pathlib import Path

from .clump import train_test_split
from .features import get_features_from_raw_features
from .features.extract import get_raw_features_for_dataset
from .labels import get_labels_for_dataset
from .netcdf import load_trajectory_dataset
from .paths import get_sosaa_dataset_paths
from .utils import hash_for_dt

MLDataset = namedtuple(
    "MLDataset",
    [
        "date",
        "paths",
        "X_raw",
        "Y_raw",
        "X_train",
        "X_valid",
        "X_test",
        "Y_train",
        "Y_valid",
        "Y_test",
        "X_scaler",
        "Y_scaler",
    ],
)


def load_and_cache_dataset(
    dt: datetime.datetime,
    clump: float,
    datasets: dict,
    input_dir: Path,
    output_dir: Path,
    progress=None,
) -> MLDataset:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    if isinstance(dt, tuple) or isinstance(dt, list):
        dt = tuple(sorted(dt))

    cached = datasets.get((dt, clump))

    if cached is not None:
        return cached

    if isinstance(dt, tuple) or isinstance(dt, list):
        mls = [
            load_and_cache_dataset(
                dtt, clump, datasets, input_dir, output_dir, progress=progress
            )
            for dtt in dt
        ]

        dp = tuple(ml.paths for ml in mls)
        X_raw = pd.concat([ml.X_raw for ml in mls], axis="index")
        Y = pd.concat([ml.Y_raw for ml in mls], axis="index")

        train_features = np.concatenate(
            [ml.X_scaler.inverse_transform(ml.X_train) for ml in mls], axis=0
        )
        train_labels = np.concatenate(
            [ml.Y_scaler.inverse_transform(ml.Y_train) for ml in mls], axis=0
        )
        valid_features = np.concatenate(
            [ml.X_scaler.inverse_transform(ml.X_valid) for ml in mls], axis=0
        )
        valid_labels = np.concatenate(
            [ml.Y_scaler.inverse_transform(ml.Y_valid) for ml in mls], axis=0
        )
        test_features = np.concatenate(
            [ml.X_scaler.inverse_transform(ml.X_test) for ml in mls], axis=0
        )
        test_labels = np.concatenate(
            [ml.Y_scaler.inverse_transform(ml.Y_test) for ml in mls], axis=0
        )
    else:
        dp = get_sosaa_dataset_paths(dt, input_dir, output_dir)
        ds = load_trajectory_dataset(dp)

        X_raw = get_raw_features_for_dataset(ds)

        X = get_features_from_raw_features(X_raw, progress=progress)
        Y = np.log10(get_labels_for_dataset(ds) + 1)

        rng = np.random.RandomState(
            seed=int.from_bytes(hash_for_dt(dt).digest(4), "little")
        )

        (
            train_features,
            test_features,
            train_labels,
            test_labels,
        ) = train_test_split(
            X,
            Y,
            test_size=0.25,
            random_state=rng,
            clump=clump,
        )
        (
            train_features,
            valid_features,
            train_labels,
            valid_labels,
        ) = train_test_split(
            train_features,
            train_labels,
            test_size=1.0 / 3.0,
            random_state=rng,
            clump=clump,
        )

        # Close the NetCDF datasets
        ds.out.close()
        ds.aer.close()
        ds.ant.close()
        ds.bio.close()
        ds.met.close()

    # Scale features to N(0,1)
    # - only fit on training data
    # - OOD inputs for constants at training time are blown up
    feature_scaler = StandardScaler().fit(train_features)
    feature_scaler.scale_[np.nonzero(feature_scaler.var_ == 0.0)] = (
        np.nan_to_num(np.inf)
    )

    label_scaler = StandardScaler().fit(train_labels)

    train_features = feature_scaler.transform(train_features)
    train_labels = label_scaler.transform(train_labels)
    valid_features = feature_scaler.transform(valid_features)
    valid_labels = label_scaler.transform(valid_labels)
    test_features = feature_scaler.transform(test_features)
    test_labels = label_scaler.transform(test_labels)

    dataset = MLDataset(
        date=dt,
        paths=dp,
        X_raw=X_raw,
        Y_raw=Y,
        X_train=train_features,
        X_valid=valid_features,
        X_test=test_features,
        Y_train=train_labels,
        Y_valid=valid_labels,
        Y_test=test_labels,
        X_scaler=feature_scaler,
        Y_scaler=label_scaler,
    )

    datasets[(dt, clump)] = dataset

    return dataset
