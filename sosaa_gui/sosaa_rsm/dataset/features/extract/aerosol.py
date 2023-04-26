from ...labels.time import get_output_time
from ...netcdf import TrajectoryDatasets


def interpolate_aerosol_emissions(ds: TrajectoryDatasets, key: str):
    import scipy as sp

    out_t = get_output_time(ds)
    out_h = ds.out["lev"][:].data

    aer_t = ds.aer["time"][:].data
    aer_h = ds.aer["mid_layer_height"][:].data

    aer_t_h = ds.aer[key][:].T

    aer_t_h_int = sp.interpolate.interp2d(
        x=aer_h,
        y=aer_t,
        z=aer_t_h,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    return aer_t_h_int(x=out_h, y=out_t)


def get_aer_emissions_features(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "time": np.repeat(get_output_time(ds), ds.out["lev"].shape[0]),
            "level": np.tile(ds.out["lev"][:].data, ds.out["time"].shape[0]),
            "aer_3_10_nm": interpolate_aerosol_emissions(
                ds, "3-10nm"
            ).flatten(),
            "aer_10_20_nm": interpolate_aerosol_emissions(
                ds, "10-20nm"
            ).flatten(),
            "aer_20_30_nm": interpolate_aerosol_emissions(
                ds, "20-30nm"
            ).flatten(),
            "aer_30_50_nm": interpolate_aerosol_emissions(
                ds, "30-50nm"
            ).flatten(),
            "aer_50_70_nm": interpolate_aerosol_emissions(
                ds, "50-70nm"
            ).flatten(),
            "aer_70_100_nm": interpolate_aerosol_emissions(
                ds, "70-100nm"
            ).flatten(),
            "aer_100_200_nm": interpolate_aerosol_emissions(
                ds, "100-200nm"
            ).flatten(),
            "aer_200_400_nm": interpolate_aerosol_emissions(
                ds, "200-400nm"
            ).flatten(),
            "aer_400_1000_nm": interpolate_aerosol_emissions(
                ds, "400-1000nm"
            ).flatten(),
        }
    ).set_index(["time", "level"])
