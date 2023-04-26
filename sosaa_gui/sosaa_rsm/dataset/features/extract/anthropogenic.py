from ...labels.time import get_output_time
from ...netcdf import TrajectoryDatasets


def interpolate_anthropogenic_emissions(ds: TrajectoryDatasets, key: str):
    import scipy as sp

    out_t = get_output_time(ds)
    out_h = ds.out["lev"][:].data

    ant_t = ds.ant["time"][:].data
    ant_h = ds.ant["mid_layer_height"][:].data

    ant_t_h = ds.ant[key][:].T

    ant_t_h_int = sp.interpolate.interp2d(
        x=ant_h,
        y=ant_t,
        z=ant_t_h,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    return ant_t_h_int(x=out_h, y=out_t)


def get_ant_emissions_features(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "time": np.repeat(get_output_time(ds), ds.out["lev"].shape[0]),
            "level": np.tile(ds.out["lev"][:].data, ds.out["time"].shape[0]),
            "ant_co": interpolate_anthropogenic_emissions(ds, "co").flatten(),
            "ant_nox": interpolate_anthropogenic_emissions(
                ds, "nox"
            ).flatten(),
            "ant_co2": interpolate_anthropogenic_emissions(
                ds, "co2"
            ).flatten(),
            "ant_nh3": interpolate_anthropogenic_emissions(
                ds, "nh3"
            ).flatten(),
            "ant_ch4": interpolate_anthropogenic_emissions(
                ds, "ch4"
            ).flatten(),
            "ant_so2": interpolate_anthropogenic_emissions(
                ds, "so2"
            ).flatten(),
            "ant_nmvoc": interpolate_anthropogenic_emissions(
                ds, "nmvoc"
            ).flatten(),
            "ant_alcohols": interpolate_anthropogenic_emissions(
                ds, "alcohols"
            ).flatten(),
            "ant_ethane": interpolate_anthropogenic_emissions(
                ds, "ethane"
            ).flatten(),
            "ant_propane": interpolate_anthropogenic_emissions(
                ds, "propane"
            ).flatten(),
            "ant_butanes": interpolate_anthropogenic_emissions(
                ds, "butanes"
            ).flatten(),
            "ant_pentanes": interpolate_anthropogenic_emissions(
                ds, "pentanes"
            ).flatten(),
            "ant_hexanes": interpolate_anthropogenic_emissions(
                ds, "hexanes"
            ).flatten(),
            "ant_ethene": interpolate_anthropogenic_emissions(
                ds, "ethene"
            ).flatten(),
            "ant_propene": interpolate_anthropogenic_emissions(
                ds, "propene"
            ).flatten(),
            "ant_acetylene": interpolate_anthropogenic_emissions(
                ds, "acetylene"
            ).flatten(),
            "ant_isoprene": interpolate_anthropogenic_emissions(
                ds, "isoprene"
            ).flatten(),
            "ant_monoterpenes": interpolate_anthropogenic_emissions(
                ds, "monoterpenes"
            ).flatten(),
            "ant_other_alkenes_and_alkynes": (
                interpolate_anthropogenic_emissions(
                    ds, "other-alkenes-and-alkynes"
                ).flatten()
            ),
            "ant_benzene": interpolate_anthropogenic_emissions(
                ds, "benzene"
            ).flatten(),
            "ant_toluene": interpolate_anthropogenic_emissions(
                ds, "toluene"
            ).flatten(),
            "ant_xylene": interpolate_anthropogenic_emissions(
                ds, "xylene"
            ).flatten(),
            "ant_trimethylbenzene": interpolate_anthropogenic_emissions(
                ds, "trimethylbenzene"
            ).flatten(),
            "ant_other_aromatics": interpolate_anthropogenic_emissions(
                ds, "other-aromatics"
            ).flatten(),
            "ant_esters": interpolate_anthropogenic_emissions(
                ds, "esters"
            ).flatten(),
            "ant_ethers": interpolate_anthropogenic_emissions(
                ds, "ethers"
            ).flatten(),
            "ant_formaldehyde": interpolate_anthropogenic_emissions(
                ds, "formaldehyde"
            ).flatten(),
            "ant_other_aldehydes": interpolate_anthropogenic_emissions(
                ds, "other-aldehydes"
            ).flatten(),
            "ant_total_ketones": interpolate_anthropogenic_emissions(
                ds, "total-ketones"
            ).flatten(),
            "ant_total_acids": interpolate_anthropogenic_emissions(
                ds, "total-acids"
            ).flatten(),
            "ant_other_vocs": interpolate_anthropogenic_emissions(
                ds, "other-VOCs"
            ).flatten(),
        }
    ).set_index(["time", "level"])
