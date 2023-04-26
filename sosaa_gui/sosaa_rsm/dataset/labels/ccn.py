from ..netcdf import TrajectoryDatasets
from .time import get_output_time


def get_ccn_concentration(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    (ccn_bin_indices,) = np.nonzero(ds.out["dp_dry_fs"][:].data > 80e-9)
    ccn_concentration = np.sum(
        ds.out["nconc_par"][:].data[:, ccn_bin_indices, :], axis=1
    )

    return pd.DataFrame(
        {
            "time": np.repeat(get_output_time(ds), ds.out["lev"].shape[0]),
            "level": np.tile(ds.out["lev"][:].data, ds.out["time"].shape[0]),
            "ccn": ccn_concentration.flatten(),
        }
    ).set_index(["time", "level"])
