from ...labels.time import get_output_time
from ...netcdf import TrajectoryDatasets


def interpolate_meteorology_values(ds: TrajectoryDatasets, key: str):
    import scipy as sp

    out_t = get_output_time(ds)
    out_h = ds.out["lev"][:].data

    met_t = ds.met["time"][:].data
    met_h = ds.met["lev"][:].data

    met_t_h = ds.met[key][:]

    met_t_h_int = sp.interpolate.interp2d(
        x=met_h,
        y=met_t,
        z=met_t_h,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    return met_t_h_int(x=out_h, y=out_t)


def interpolate_meteorology_time_values(ds: TrajectoryDatasets, key: str):
    import numpy as np
    import scipy as sp

    out_t = get_output_time(ds)
    out_h = ds.out["lev"][:].data

    met_t = ds.met["time"][:].data

    met_t_v = ds.met[key][:]

    met_t_int = sp.interpolate.interp1d(
        x=met_t,
        y=met_t_v,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    return np.repeat(
        met_t_int(x=out_t).reshape(-1, 1),
        out_h.shape[0],
        axis=1,
    )


def get_meteorology_features(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "time": np.repeat(get_output_time(ds), ds.out["lev"].shape[0]),
            "level": np.tile(ds.out["lev"][:].data, ds.out["time"].shape[0]),
            "met_t": interpolate_meteorology_values(ds, "t").flatten(),
            "met_q": interpolate_meteorology_values(ds, "q").flatten(),
            "met_ssr": interpolate_meteorology_time_values(
                ds, "ssr"
            ).flatten(),
            "met_lsm": interpolate_meteorology_time_values(
                ds, "lsm"
            ).flatten(),
            "met_blh": interpolate_meteorology_time_values(
                ds, "blh"
            ).flatten(),
        }
    ).set_index(["time", "level"])
