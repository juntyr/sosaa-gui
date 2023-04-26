from ...labels.time import get_output_time
from ...netcdf import TrajectoryDatasets


def interpolate_biogenic_emissions(ds: TrajectoryDatasets, key: str):
    import numpy as np
    import scipy as sp

    out_t = get_output_time(ds)
    out_h = ds.out["lev"][:].data

    # depth of each box layer, assuming level heights are midpoints
    #  and end points are clamped
    out_d = (
        np.array(list(out_h[1:]) + [out_h[-1]])
        - np.array([out_h[0]] + list(out_h[:-1]))
    ) / 2.0

    bio_t = ds.bio["time"][:].data

    # Biogenic emissions are limited to boxes at <= 10m height
    biogenic_emission_layers = np.nonzero(out_h <= 10.0)
    biogenic_emission_layer_height_cumsum = np.cumsum(
        out_d[biogenic_emission_layers]
    )
    biogenic_emission_layer_proportion = (
        biogenic_emission_layer_height_cumsum
        / biogenic_emission_layer_height_cumsum[-1]
    )
    num_biogenic_emission_layers = sum(out_h <= 10.0)

    bio_t_h = np.zeros(shape=(out_t.size, out_h.size))

    bio_t_int = sp.interpolate.interp1d(
        x=bio_t,
        y=ds.bio[key][:],
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Split up the biogenic emissions relative to the depth of the boxes
    bio_t_h[:, biogenic_emission_layers] = (
        np.tile(bio_t_int(x=out_t), (num_biogenic_emission_layers, 1, 1))
        * biogenic_emission_layer_proportion.reshape(-1, 1, 1)
    ).T

    return bio_t_h


def get_bio_emissions_features(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "time": np.repeat(get_output_time(ds), ds.out["lev"].shape[0]),
            "level": np.tile(ds.out["lev"][:].data, ds.out["time"].shape[0]),
            "bio_acetaldehyde": interpolate_biogenic_emissions(
                ds, "acetaldehyde"
            ).flatten(),
            "bio_acetone": interpolate_biogenic_emissions(
                ds, "acetone"
            ).flatten(),
            "bio_butanes_and_higher_alkanes": interpolate_biogenic_emissions(
                ds, "butanes-and-higher-alkanes"
            ).flatten(),
            "bio_butanes_and_higher_alkenes": interpolate_biogenic_emissions(
                ds, "butenes-and-higher-alkenes"
            ).flatten(),
            "bio_ch4": interpolate_biogenic_emissions(ds, "CH4").flatten(),
            "bio_co": interpolate_biogenic_emissions(ds, "CO").flatten(),
            "bio_ethane": interpolate_biogenic_emissions(
                ds, "ethane"
            ).flatten(),
            "bio_ethanol": interpolate_biogenic_emissions(
                ds, "ethanol"
            ).flatten(),
            "bio_ethene": interpolate_biogenic_emissions(
                ds, "ethene"
            ).flatten(),
            "bio_formaldehyde": interpolate_biogenic_emissions(
                ds, "formaldehyde"
            ).flatten(),
            "bio_hydrogen_cyanide": interpolate_biogenic_emissions(
                ds, "hydrogen-cyanide"
            ).flatten(),
            "bio_iosprene": interpolate_biogenic_emissions(
                ds, "isoprene"
            ).flatten(),
            "bio_mbo": interpolate_biogenic_emissions(ds, "MBO").flatten(),
            "bio_methanol": interpolate_biogenic_emissions(
                ds, "methanol"
            ).flatten(),
            "bio_methyl_bromide": interpolate_biogenic_emissions(
                ds, "methyl-bromide"
            ).flatten(),
            "bio_methyl_chloride": interpolate_biogenic_emissions(
                ds, "methyl-chloride"
            ).flatten(),
            "bio_methyl_iodide": interpolate_biogenic_emissions(
                ds, "methyl-iodide"
            ).flatten(),
            "bio_other_aldehydes": interpolate_biogenic_emissions(
                ds, "other-aldehydes"
            ).flatten(),
            "bio_other_ketones": interpolate_biogenic_emissions(
                ds, "other-ketones"
            ).flatten(),
            "bio_other_monoterpenes": interpolate_biogenic_emissions(
                ds, "other-monoterpenes"
            ).flatten(),
            "bio_pinene_a": interpolate_biogenic_emissions(
                ds, "pinene-a"
            ).flatten(),
            "bio_pinene_b": interpolate_biogenic_emissions(
                ds, "pinene-b"
            ).flatten(),
            "bio_propane": interpolate_biogenic_emissions(
                ds, "propane"
            ).flatten(),
            "bio_propene": interpolate_biogenic_emissions(
                ds, "propene"
            ).flatten(),
            "bio_sesquiterpenes": interpolate_biogenic_emissions(
                ds, "sesquiterpenes"
            ).flatten(),
            "bio_toluene": interpolate_biogenic_emissions(
                ds, "toluene"
            ).flatten(),
            "bio_ch2br2": interpolate_biogenic_emissions(
                ds, "CH2Br2"
            ).flatten(),
            "bio_ch3i": interpolate_biogenic_emissions(ds, "CH3I").flatten(),
            "bio_chbr3": interpolate_biogenic_emissions(ds, "CHBr3").flatten(),
            "bio_dms": interpolate_biogenic_emissions(ds, "DMS").flatten(),
        }
    ).set_index(["time", "level"])
