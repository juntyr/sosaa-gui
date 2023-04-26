from ..expand import (
    generate_time_level_windows,
    generate_windowed_feature_names,
    time_level_window_mean,
)
from ..utils import df_to_numpy


def get_features_from_raw_features(raw_features, progress=None):
    import joblib
    import numpy as np
    import pandas as pd

    raw_features_np = df_to_numpy(raw_features)

    if progress is not None:
        progress.update_minor(
            value=0,
            min=0,
            max=len(generate_time_level_windows()),
            format="Expanding space-time %v/%m",
        )

    features_np = np.concatenate(
        [
            raw_features.index.get_level_values(0)
            .to_numpy()
            .reshape(
                (
                    raw_features.index.levels[0].size,
                    raw_features.index.levels[1].size,
                    1,
                )
            ),
            raw_features.index.get_level_values(1)
            .to_numpy()
            .reshape(
                (
                    raw_features.index.levels[0].size,
                    raw_features.index.levels[1].size,
                    1,
                )
            ),
        ]
        + joblib.Parallel(n_jobs=-1, prefer="threads")(
            [
                joblib.delayed(time_level_window_mean)(
                    raw_features_np, t, l, progress=progress
                )
                for t, l in generate_time_level_windows()
            ]
        ),
        axis=2,
    )

    # Trim off the first two days, for which the time features are ill-defined
    features_np_trimmed = features_np[95:-1, :, :]

    feature_names = ["time", "level"] + generate_windowed_feature_names(
        raw_features.columns
    )

    features = pd.DataFrame(
        features_np_trimmed.reshape(
            features_np_trimmed.shape[0] * features_np_trimmed.shape[1],
            features_np_trimmed.shape[2],
        ),
        columns=feature_names,
    ).set_index(["time", "level"])

    return features
