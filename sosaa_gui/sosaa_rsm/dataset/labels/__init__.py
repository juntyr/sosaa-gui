from ..netcdf import TrajectoryDatasets
from ..utils import df_to_numpy
from .ccn import get_ccn_concentration


def get_labels_for_dataset(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    ccn_concentration = get_ccn_concentration(ds)

    ccn_concentration_np = df_to_numpy(ccn_concentration)

    labels_np = np.concatenate(
        [
            ccn_concentration.index.get_level_values(0)
            .to_numpy()
            .reshape(
                (
                    ccn_concentration.index.levels[0].size,
                    ccn_concentration.index.levels[1].size,
                    1,
                )
            ),
            ccn_concentration.index.get_level_values(1)
            .to_numpy()
            .reshape(
                (
                    ccn_concentration.index.levels[0].size,
                    ccn_concentration.index.levels[1].size,
                    1,
                )
            ),
            ccn_concentration_np.reshape(
                (
                    ccn_concentration_np.shape[0],
                    ccn_concentration_np.shape[1],
                    1,
                )
            ),
        ],
        axis=2,
    )

    # Trim off the first two days, for which the time features are ill-defined
    labels_np_trimmed = labels_np[96:, :, :]

    label_names = ["time", "level", "ccn"]

    labels = pd.DataFrame(
        labels_np_trimmed.reshape(
            labels_np_trimmed.shape[0] * labels_np_trimmed.shape[1],
            labels_np_trimmed.shape[2],
        ),
        columns=label_names,
    ).set_index(["time", "level"])

    return labels
