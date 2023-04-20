from __future__ import annotations

import abc
import datetime
import hashlib
import itertools
from collections import namedtuple
from functools import partial
from pathlib import Path

TrajectoryPaths = namedtuple(
    "TrajectoryPaths",
    [
        "date",
        "out",
        "aer",
        "ant",
        "bio",
        "met",
    ],
)
TrajectoryDatasets = namedtuple(
    "TrajectoryDatasets",
    [
        "date",
        "out",
        "aer",
        "ant",
        "bio",
        "met",
    ],
)
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
IcarusPrediction = namedtuple(
    "IcarusPrediction",
    [
        "prediction",
        "uncertainty",
        "confidence",
    ],
)
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


def get_sosaa_dataset_paths(
    dt: datetime.datetime, input_dir: Path, output_dir: Path
) -> TrajectoryPaths:
    out_path = output_dir / "output.nc"
    aer_path = (
        input_dir
        / f"{dt.strftime('%Y%m%d')}_7daybwd_Hyde_traj_AER_{24-dt.hour:02}_L3.nc"
    )
    ant_path = (
        input_dir
        / f"{dt.strftime('%Y%m%d')}_7daybwd_Hyde_traj_ANT_{24-dt.hour:02}_L3.nc"
    )
    bio_path = (
        input_dir
        / f"{dt.strftime('%Y%m%d')}_7daybwd_Hyde_traj_BIO_{24-dt.hour:02}_L3.nc"
    )
    met_path = input_dir / f"METEO_{dt.strftime('%Y%m%d')}_R{24-dt.hour:02}.nc"

    out_path = out_path.resolve()
    aer_path = aer_path.resolve()
    ant_path = ant_path.resolve()
    bio_path = bio_path.resolve()
    met_path = met_path.resolve()

    for file in [out_path, aer_path, ant_path, bio_path, met_path]:
        if not file.exists():
            raise FileNotFoundError(str(file))

    return TrajectoryPaths(
        date=dt,
        out=out_path,
        aer=aer_path,
        ant=ant_path,
        bio=bio_path,
        met=met_path,
    )


def load_trajectory_dataset(paths: TrajectoryPaths) -> TrajectoryDatasets:
    from netCDF4 import Dataset

    outds = Dataset(paths.out, "r", format="NETCDF4")
    aerds = Dataset(paths.aer, "r", format="NETCDF4")
    antds = Dataset(paths.ant, "r", format="NETCDF4")
    biods = Dataset(paths.bio, "r", format="NETCDF4")
    metds = Dataset(paths.met, "r", format="NETCDF4")

    return TrajectoryDatasets(
        date=paths.date,
        out=outds,
        aer=aerds,
        ant=antds,
        bio=biods,
        met=metds,
    )


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


def get_output_time(ds: TrajectoryDatasets):
    fdom = datetime.datetime.strptime(
        ds.out["time"].__dict__["first_day_of_month"],
        "%Y-%m-%d %H:%M:%S",
    )
    dt = (ds.date - fdom).total_seconds()

    out_t = ds.out["time"][:].data

    return out_t - dt


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


def interpolate_biogenic_emissions(ds: TrajectoryDatasets, key: str):
    import numpy as np
    import scipy as sp

    out_t = get_output_time(ds)
    out_h = ds.out["lev"][:].data

    # depth of each box layer, assuming level heights are midpoints and end points are clamped
    out_d = (
        np.array(list(out_h[1:]) + [out_h[-1]])
        - np.array([out_h[0]] + list(out_h[:-1]))
    ) / 2.0

    bio_t = ds.bio["time"][:].data

    # Biogenic emissions are limited to boxes at <= 10m height
    biogenic_emission_layers = np.nonzero(out_h <= 10.0)
    biogenic_emission_layer_height_cumsum = np.cumsum(out_d[biogenic_emission_layers])
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


def get_meteorology_features(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "time": np.repeat(get_output_time(ds), ds.out["lev"].shape[0]),
            "level": np.tile(ds.out["lev"][:].data, ds.out["time"].shape[0]),
            "met_t": interpolate_meteorology_values(ds, "t").flatten(),
            "met_q": interpolate_meteorology_values(ds, "q").flatten(),
            "met_ssr": interpolate_meteorology_time_values(ds, "ssr").flatten(),
            "met_lsm": interpolate_meteorology_time_values(ds, "lsm").flatten(),
            "met_blh": interpolate_meteorology_time_values(ds, "blh").flatten(),
        }
    ).set_index(["time", "level"])


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
            "bio_acetone": interpolate_biogenic_emissions(ds, "acetone").flatten(),
            "bio_butanes_and_higher_alkanes": interpolate_biogenic_emissions(
                ds, "butanes-and-higher-alkanes"
            ).flatten(),
            "bio_butanes_and_higher_alkenes": interpolate_biogenic_emissions(
                ds, "butenes-and-higher-alkenes"
            ).flatten(),
            "bio_ch4": interpolate_biogenic_emissions(ds, "CH4").flatten(),
            "bio_co": interpolate_biogenic_emissions(ds, "CO").flatten(),
            "bio_ethane": interpolate_biogenic_emissions(ds, "ethane").flatten(),
            "bio_ethanol": interpolate_biogenic_emissions(ds, "ethanol").flatten(),
            "bio_ethene": interpolate_biogenic_emissions(ds, "ethene").flatten(),
            "bio_formaldehyde": interpolate_biogenic_emissions(
                ds, "formaldehyde"
            ).flatten(),
            "bio_hydrogen_cyanide": interpolate_biogenic_emissions(
                ds, "hydrogen-cyanide"
            ).flatten(),
            "bio_iosprene": interpolate_biogenic_emissions(ds, "isoprene").flatten(),
            "bio_mbo": interpolate_biogenic_emissions(ds, "MBO").flatten(),
            "bio_methanol": interpolate_biogenic_emissions(ds, "methanol").flatten(),
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
            "bio_pinene_a": interpolate_biogenic_emissions(ds, "pinene-a").flatten(),
            "bio_pinene_b": interpolate_biogenic_emissions(ds, "pinene-b").flatten(),
            "bio_propane": interpolate_biogenic_emissions(ds, "propane").flatten(),
            "bio_propene": interpolate_biogenic_emissions(ds, "propene").flatten(),
            "bio_sesquiterpenes": interpolate_biogenic_emissions(
                ds, "sesquiterpenes"
            ).flatten(),
            "bio_toluene": interpolate_biogenic_emissions(ds, "toluene").flatten(),
            "bio_ch2br2": interpolate_biogenic_emissions(ds, "CH2Br2").flatten(),
            "bio_ch3i": interpolate_biogenic_emissions(ds, "CH3I").flatten(),
            "bio_chbr3": interpolate_biogenic_emissions(ds, "CHBr3").flatten(),
            "bio_dms": interpolate_biogenic_emissions(ds, "DMS").flatten(),
        }
    ).set_index(["time", "level"])


def get_aer_emissions_features(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "time": np.repeat(get_output_time(ds), ds.out["lev"].shape[0]),
            "level": np.tile(ds.out["lev"][:].data, ds.out["time"].shape[0]),
            "aer_3_10_nm": interpolate_aerosol_emissions(ds, "3-10nm").flatten(),
            "aer_10_20_nm": interpolate_aerosol_emissions(ds, "10-20nm").flatten(),
            "aer_20_30_nm": interpolate_aerosol_emissions(ds, "20-30nm").flatten(),
            "aer_30_50_nm": interpolate_aerosol_emissions(ds, "30-50nm").flatten(),
            "aer_50_70_nm": interpolate_aerosol_emissions(ds, "50-70nm").flatten(),
            "aer_70_100_nm": interpolate_aerosol_emissions(ds, "70-100nm").flatten(),
            "aer_100_200_nm": interpolate_aerosol_emissions(ds, "100-200nm").flatten(),
            "aer_200_400_nm": interpolate_aerosol_emissions(ds, "200-400nm").flatten(),
            "aer_400_1000_nm": interpolate_aerosol_emissions(
                ds, "400-1000nm"
            ).flatten(),
        }
    ).set_index(["time", "level"])


def get_ant_emissions_features(ds: TrajectoryDatasets):
    import numpy as np
    import pandas as pd

    return pd.DataFrame(
        {
            "time": np.repeat(get_output_time(ds), ds.out["lev"].shape[0]),
            "level": np.tile(ds.out["lev"][:].data, ds.out["time"].shape[0]),
            "ant_co": interpolate_anthropogenic_emissions(ds, "co").flatten(),
            "ant_nox": interpolate_anthropogenic_emissions(ds, "nox").flatten(),
            "ant_co2": interpolate_anthropogenic_emissions(ds, "co2").flatten(),
            "ant_nh3": interpolate_anthropogenic_emissions(ds, "nh3").flatten(),
            "ant_ch4": interpolate_anthropogenic_emissions(ds, "ch4").flatten(),
            "ant_so2": interpolate_anthropogenic_emissions(ds, "so2").flatten(),
            "ant_nmvoc": interpolate_anthropogenic_emissions(ds, "nmvoc").flatten(),
            "ant_alcohols": interpolate_anthropogenic_emissions(
                ds, "alcohols"
            ).flatten(),
            "ant_ethane": interpolate_anthropogenic_emissions(ds, "ethane").flatten(),
            "ant_propane": interpolate_anthropogenic_emissions(ds, "propane").flatten(),
            "ant_butanes": interpolate_anthropogenic_emissions(ds, "butanes").flatten(),
            "ant_pentanes": interpolate_anthropogenic_emissions(
                ds, "pentanes"
            ).flatten(),
            "ant_hexanes": interpolate_anthropogenic_emissions(ds, "hexanes").flatten(),
            "ant_ethene": interpolate_anthropogenic_emissions(ds, "ethene").flatten(),
            "ant_propene": interpolate_anthropogenic_emissions(ds, "propene").flatten(),
            "ant_acetylene": interpolate_anthropogenic_emissions(
                ds, "acetylene"
            ).flatten(),
            "ant_isoprene": interpolate_anthropogenic_emissions(
                ds, "isoprene"
            ).flatten(),
            "ant_monoterpenes": interpolate_anthropogenic_emissions(
                ds, "monoterpenes"
            ).flatten(),
            "ant_other_alkenes_and_alkynes": interpolate_anthropogenic_emissions(
                ds, "other-alkenes-and-alkynes"
            ).flatten(),
            "ant_benzene": interpolate_anthropogenic_emissions(ds, "benzene").flatten(),
            "ant_toluene": interpolate_anthropogenic_emissions(ds, "toluene").flatten(),
            "ant_xylene": interpolate_anthropogenic_emissions(ds, "xylene").flatten(),
            "ant_trimethylbenzene": interpolate_anthropogenic_emissions(
                ds, "trimethylbenzene"
            ).flatten(),
            "ant_other_aromatics": interpolate_anthropogenic_emissions(
                ds, "other-aromatics"
            ).flatten(),
            "ant_esters": interpolate_anthropogenic_emissions(ds, "esters").flatten(),
            "ant_ethers": interpolate_anthropogenic_emissions(ds, "ethers").flatten(),
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


# https://stackoverflow.com/a/67809235
def df_to_numpy(df):
    try:
        shape = [len(level) for level in df.index.levels]
    except AttributeError:
        shape = [len(df.index)]
    ncol = df.shape[-1]
    if ncol > 1:
        shape.append(ncol)
    return df.to_numpy().reshape(shape)


def generate_time_level_windows():
    # -0.5h, -1.5h, -3h, -6h, -12h, -24h, -48h
    # 0, -2, -5, -11, -23, -47, -95
    time_windows = [
        (0, 0),
        (-2, -1),
        (-5, -3),
        (-11, -6),
        (-23, -12),
        (-47, -24),
        (-95, -48),
    ]

    # +1l, +2l, +4l, +8l, +16l, +32l, +64
    top_windows = [(1, 1), (1, 2), (1, 4), (2, 8), (2, 16), (3, 32), (3, 64)]
    mid_windows = [(0, 0), (0, 0), (0, 0), (-1, 1), (-1, 1), (-2, 2), (-2, 2)]
    bot_windows = [
        (-1, -1),
        (-2, -1),
        (-4, -1),
        (-8, -2),
        (-16, -2),
        (-32, -3),
        (-64, -3),
    ]

    return list(
        itertools.chain(
            zip(time_windows, top_windows),
            zip(time_windows, mid_windows),
            zip(time_windows, bot_windows),
        )
    )


def generate_windowed_feature_names(columns):
    time_windows = ["-0.5h", "-1.5h", "-3h", "-6h", "-12h", "-24h", "-48h"]

    top_windows = ["+1l", "+2l", "+4l", "+8l", "+16l", "+32l", "+64l"]
    mid_windows = ["+0l", "+0l", "+0l", "±1l", "±1l", "±2l", "±2l"]
    bot_windows = ["-1l", "-2l", "-4l", "-8l", "-16l", "-32l", "-64l"]

    names = []

    for t, l in itertools.chain(
        zip(time_windows, top_windows),
        zip(time_windows, mid_windows),
        zip(time_windows, bot_windows),
    ):
        for c in columns:
            names.append(f"{c}{t}{l}")

    return names


def time_level_window_mean(input, t_range, l_range, progress=None):
    import numpy as np

    output = np.zeros(shape=input.shape)

    for t in range(input.shape[0]):
        mint = min(max(0, t + t_range[0]), input.shape[0])
        maxt = max(0, min(t + 1 + t_range[1], input.shape[0]))

        if mint == maxt:
            continue

        for l in range(input.shape[1]):
            minl = min(max(0, l + l_range[0]), input.shape[1])
            maxl = max(0, min(l + 1 + l_range[1], input.shape[1]))

            if minl == maxl:
                continue

            output[t, l, :] = np.mean(input[mint:maxt, minl:maxl, :], axis=(0, 1))

    if progress is not None:
        progress.update_minor()

    return output


def get_raw_features_for_dataset(ds: TrajectoryDatasets):
    import pandas as pd

    bio_features = get_bio_emissions_features(ds)
    aer_features = get_aer_emissions_features(ds) * 1e21
    ant_features = get_ant_emissions_features(ds)
    met_features = get_meteorology_features(ds)

    return pd.concat(
        [
            bio_features,
            aer_features,
            ant_features,
            met_features,
        ],
        axis="columns",
    )


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
                (ccn_concentration_np.shape[0], ccn_concentration_np.shape[1], 1)
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


def hash_for_dt(dt):
    if not (isinstance(dt, tuple) or isinstance(dt, list)):
        dt = [dt]

    dt_str = ".".join(dtt.strftime("%d.%m.%Y-%H:00%z") for dtt in dt)

    h = hashlib.shake_256()
    h.update(dt_str.encode("ascii"))

    return h


"""
Clumped 0/1 sampler using a Markov Process

P(0) = p and P(1) = 1-p
clump = 0 => IID samples
clump -> 1 => highly correlated samples

"""


class Clump:
    def __init__(self, p=0.5, clump=0.0, rng=None):
        import numpy as np

        a = 1 - (1 - p) * (1 - clump)
        b = (1 - a) * p / (1 - p)

        self.C = np.array([[a, 1 - a], [b, 1 - b]])

        self.i = 0 if rng.random() < p else 1

    def sample(self, rng):
        p = self.C[self.i, 0]
        u = rng.random()

        self.i = 0 if u < p else 1

        return self.i

    def steady(self, X):
        import numpy as np

        return np.matmul(X, self.C)


def train_test_split(X, Y, test_size=0.25, random_state=None, shuffle=True, clump=0.0):
    import numpy as np
    import pandas as pd

    assert len(X) == len(Y)
    assert type(X) == type(Y)
    assert test_size > 0.0
    assert test_size < 1.0
    assert random_state is not None
    assert clump >= 0.0
    assert clump < 1.0

    c = Clump(p=test_size, clump=clump, rng=random_state)

    if isinstance(X, pd.DataFrame):
        assert X.index.values.shape == Y.index.values.shape

        # Split only based on the first-level index instead of flattening
        n1 = len(X.index.levels[1])
        n0 = len(X) // n1

        C = np.array([c.sample(random_state) for _ in range(n0)])
        (I_train,) = np.nonzero(C)
        I_train = np.repeat(I_train, n1) * n1 + np.tile(np.arange(n1), len(I_train))
        (I_test,) = np.nonzero(1 - C)
        I_test = np.repeat(I_test, n1) * n1 + np.tile(np.arange(n1), len(I_test))
    else:
        C = np.array([c.sample(random_state) for _ in range(len(X))])
        (I_train,) = np.nonzero(C)
        (I_test,) = np.nonzero(1 - C)

    if shuffle:
        random_state.shuffle(I_train)
        random_state.shuffle(I_test)

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[I_train]
        X_test = X.iloc[I_test]

        Y_train = Y.iloc[I_train]
        Y_test = Y.iloc[I_test]
    else:
        X_train = X[I_train]
        X_test = X[I_test]

        Y_train = Y[I_train]
        Y_test = Y[I_test]

    return X_train, X_test, Y_train, Y_test


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

        train_features, test_features, train_labels, test_labels = train_test_split(
            X,
            Y,
            test_size=0.25,
            random_state=rng,
            clump=clump,
        )
        train_features, valid_features, train_labels, valid_labels = train_test_split(
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
    feature_scaler.scale_[np.nonzero(feature_scaler.var_ == 0.0)] = np.nan_to_num(
        np.inf
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


class RandomForestSosaaRSM(IcarusRSM):
    def fit(
        self,
        X_train,  #: np.ndarray,
        Y_train,  #: np.ndarray,
        X_valid,  #: np.ndarray,
        Y_valid,  #: np.ndarray,
        rng,  #: np.random.Generator,
        n_trees: int = 16,
        progress=None,
    ) -> RandomForestSosaaRSM:
        import numpy as np
        from sklearn.covariance import EmpiricalCovariance
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestRegressor

        # Fake use Y_valid
        Y_valid = Y_valid

        assert Y_train.shape[1:] == (1,)

        if progress is not None:
            progress.update_minor(
                value=0, min=0, max=5, format="Fitting the Truncated PCA OOD Detector"
            )

        self.pca = PCA(random_state=rng).fit(X_train)
        self.bn = np.searchsorted(np.cumsum(self.pca.explained_variance_ratio_), 0.95)

        if progress is not None:
            progress.update_minor(format="Fitting Auto-Associative Error Covariance")

        self.cov = EmpiricalCovariance().fit(
            (self._predict_truncated_pca(X_train) - X_train)
        )

        self.err_valid = np.sort(
            self.cov.mahalanobis(self._predict_truncated_pca(X_valid) - X_valid)
        )

        if progress is not None:
            progress.update_minor(
                format="Training the Prediction Model and Uncertainty Quantifier"
            )

        self.predictor = RandomForestRegressor(
            n_estimators=n_trees,
            random_state=rng,
            n_jobs=-1,
            min_samples_leaf=5,
            max_features=1.0 / 3.0,
        ).fit(X_train, Y_train.ravel())

        if progress is not None:
            progress.update_minor(format="Finished Training the SOSAA RSM")

        return self

    def predict(
        self,
        X_test,  #: np.ndarray,
        rng,  #: np.random.Generator,
        progress=None,
    ) -> IcarusPrediction:
        import joblib
        import numpy as np

        # No extra randomness is needed during prediction
        rng = rng

        if progress is not None:
            progress.update_minor(
                value=0,
                min=0,
                max=(1 + len(self.predictor.estimators_)),
                format="Generating Confidence Scores",
            )

        confidence = 1.0 - np.searchsorted(
            self.err_valid,
            self.cov.mahalanobis((self._predict_truncated_pca(X_test) - X_test)),
        ) / len(self.err_valid)

        if progress is not None:
            progress.update_minor(format="Generating %v/%m Ensemble Predictions")

        def tree_predict(tree, X_test, progress=None) -> np.ndarray:
            if progress is not None:
                progress.update_minor()

            return tree.predict(X_test)

        predictions = joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(tree_predict)(tree, X_test, progress)
            for tree in self.predictor.estimators_
        )

        prediction = np.mean(np.stack(predictions, axis=0), axis=0).reshape(
            (len(X_test), 1)
        )
        uncertainty = np.std(np.stack(predictions, axis=0), axis=0).reshape(
            (len(X_test), 1)
        )

        return IcarusPrediction(
            prediction=prediction,
            uncertainty=uncertainty,
            confidence=confidence,
        )

    def _predict_truncated_pca(
        self,
        X,  #: np.ndarray
    ):  # -> np.ndarray:
        import numpy as np

        if self.pca.mean_ is not None:
            X = X - self.pca.mean_

        X_trans = np.dot(X, self.pca.components_[: self.bn].T)
        X = np.dot(X_trans, self.pca.components_[: self.bn])

        if self.pca.mean_ is not None:
            X = X + self.pca.mean_

        return X


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


def analyse_icarus_predictions(
    predictions: IcarusPrediction,
    analysis,  #: Callable[[np.ndarray, np.ndarray, np.random.Generator, dict], np.ndarray],
    rng,  #: np.random.Generator,
    n_uncertain_samples: int = 1,  # number of samples to draw from expand each prediction per run
    n_analysis_runs: int = 100,  # number of repeats of the analysis to gather uncertainty
    **kwargs,
):
    import numpy as np

    progress = kwargs.get("progress", None)

    if progress is not None:
        progress.update_minor(
            value=0, min=0, max=n_analysis_runs, format="Monte Carlo Analysis Run %v/%m"
        )

    confidence = np.mean(predictions.confidence)

    results = []

    for _ in range(n_analysis_runs):
        confs = []
        preds = []
        for _ in range(n_uncertain_samples):
            I_conf = (
                rng.random(size=predictions.confidence.shape) <= predictions.confidence
            )
            (I_conf,) = np.nonzero(I_conf)

            confs.append(I_conf)
            preds.append(
                rng.normal(
                    loc=predictions.prediction[I_conf],
                    scale=predictions.uncertainty[I_conf],
                )
            )
        confs = np.concatenate(confs, axis=0)
        preds = np.concatenate(preds, axis=0)

        results.append(analysis(preds, confs, rng, **kwargs))

        if progress is not None:
            progress.update_minor()

    prediction = np.mean(np.stack(results, axis=0), axis=0)
    uncertainty = np.std(np.stack(results, axis=0), axis=0)

    return IcarusPrediction(
        prediction=prediction,
        uncertainty=uncertainty,
        confidence=confidence,
    )


def calculate_calibration_error(
    Y_true,  #: np.ndarray,
    Y_pred,  #: np.ndarray,
    Y_stdv,  #: np.ndarray,
    Y_conf,  #: np.ndarray,
    N: int = 1000,
    progress=None,
) -> IcarusPrediction:
    import numpy as np
    import scipy as sp

    sce = 0.0

    if progress is not None:
        progress.update_minor(value=0, min=0, max=N, format="Checking Percentile %p")

    for i in range(N + 1):
        if progress is not None:
            progress.update_minor()

        p = i / N

        Yp = Y_pred.flatten() + Y_stdv.flatten() * sp.stats.norm.ppf(p)
        sce += (p - np.average(Y_true.flatten() < Yp, weights=Y_conf)) ** 2

    rmsce = np.sqrt(sce / N)

    return IcarusPrediction(
        prediction=rmsce,
        uncertainty=None,
        confidence=np.mean(Y_conf),
    )


def analyse_train_test_perforance(
    model: IcarusRSM,
    dataset: MLDataset,
    rng,  #: np.random.Generator,
    n_samples: int,
    progress=None,
    **kwargs,
) -> TrainTestEvaluation:
    import numpy as np

    def mse_mae_analysis(Y_true, Y_pred, I_pred, rng, **kwargs):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        Y_true = dataset.Y_scaler.inverse_transform(Y_true[I_pred])
        Y_pred = dataset.Y_scaler.inverse_transform(Y_pred)

        mse = mean_squared_error(Y_true, Y_pred)
        mae = mean_absolute_error(Y_true, Y_pred)
        r2 = r2_score(Y_true, Y_pred)

        return np.array([mse, mae, r2])

    train_predictions = []
    for i in range(n_samples):
        if progress is not None:
            progress.update_major(
                format=f"Predicting on the Training Dataset {i}/{n_samples}"
            )

        train_predictions.append(
            model.predict(dataset.X_train, rng, progress=progress, **kwargs)
        )

    if progress is not None:
        progress.update_major(
            format=f"Combining the Predictions on the Training Dataset"
        )
        progress.update_minor(
            value=0,
            min=0,
            max=len(dataset.X_train),
            format="Training Prediction %v/%m",
        )

    combined_train_predictions = IcarusPrediction(
        prediction=[],
        uncertainty=[],
        confidence=[],
    )

    for i in range(len(dataset.X_train)):
        predictions = np.array([p.prediction[i] for p in train_predictions])
        uncertainties = np.array([p.uncertainty[i] for p in train_predictions])
        confidences = np.array([p.confidence[i] for p in train_predictions])

        def combine_predictions(Y_pred, I_pred, rng, **kwargs):
            return (
                np.mean(Y_pred)
                if len(Y_pred) > 0
                else np.mean(
                    train_predictions[rng.choice(len(train_predictions))].prediction[i]
                )
            )

        cp = analyse_icarus_predictions(
            IcarusPrediction(
                prediction=predictions,
                uncertainty=uncertainties,
                confidence=confidences,
            ),
            combine_predictions,
            rng,
            n_uncertain_samples=1,
            n_analysis_runs=10,
            progress=None,
        )

        combined_train_predictions.prediction.append(cp.prediction)
        combined_train_predictions.uncertainty.append(cp.uncertainty)
        combined_train_predictions.confidence.append(cp.confidence)

        if progress is not None:
            progress.update_minor()

    train_predictions = IcarusPrediction(
        prediction=np.array(combined_train_predictions.prediction).reshape(-1, 1),
        uncertainty=np.array(combined_train_predictions.uncertainty).reshape(-1, 1),
        confidence=np.array(combined_train_predictions.confidence),
    )

    if progress is not None:
        progress.update_major(format="Evaluating on the Training Dataset")

    train_eval = analyse_icarus_predictions(
        train_predictions,
        partial(mse_mae_analysis, dataset.Y_train),
        rng,
        n_uncertain_samples=1,
        n_analysis_runs=10,
        progress=progress,
    )

    if progress is not None:
        progress.update_major(format="Calculating the Training Calibration Error")

    train_rmsce = calculate_calibration_error(
        dataset.Y_scaler.inverse_transform(dataset.Y_train),
        dataset.Y_scaler.inverse_transform(train_predictions.prediction),
        train_predictions.uncertainty * dataset.Y_scaler.scale_,
        train_predictions.confidence,
        progress=progress,
    )

    test_predictions = []
    for i in range(n_samples):
        if progress is not None:
            progress.update_major(
                format=f"Predicting on the Test Dataset {i}/{n_samples}"
            )

        test_predictions.append(
            model.predict(dataset.X_test, rng, progress=progress, **kwargs)
        )

    if progress is not None:
        progress.update_major(format=f"Combining the Predictions on the Test Dataset")

        progress.update_minor(
            value=0,
            min=0,
            max=len(dataset.X_test),
            format="Test Prediction %v/%m",
        )

    combined_test_predictions = IcarusPrediction(
        prediction=[],
        uncertainty=[],
        confidence=[],
    )

    for i in range(len(dataset.X_test)):
        predictions = np.array([p.prediction[i] for p in test_predictions])
        uncertainties = np.array([p.uncertainty[i] for p in test_predictions])
        confidences = np.array([p.confidence[i] for p in test_predictions])

        def combine_predictions(Y_pred, I_pred, rng, **kwargs):
            return (
                np.mean(Y_pred)
                if len(Y_pred) > 0
                else np.mean(
                    test_predictions[rng.choice(len(test_predictions))].prediction[i]
                )
            )

        cp = analyse_icarus_predictions(
            IcarusPrediction(
                prediction=predictions,
                uncertainty=uncertainties,
                confidence=confidences,
            ),
            combine_predictions,
            rng,
            n_uncertain_samples=1,
            n_analysis_runs=10,
            progress=None,
        )

        combined_test_predictions.prediction.append(cp.prediction)
        combined_test_predictions.uncertainty.append(cp.uncertainty)
        combined_test_predictions.confidence.append(cp.confidence)

        if progress is not None:
            progress.update_minor()

    test_predictions = IcarusPrediction(
        prediction=np.array(combined_test_predictions.prediction).reshape(-1, 1),
        uncertainty=np.array(combined_test_predictions.uncertainty).reshape(-1, 1),
        confidence=np.array(combined_test_predictions.confidence),
    )

    if progress is not None:
        progress.update_major(format="Evaluating on the Test Dataset")

    test_eval = analyse_icarus_predictions(
        test_predictions,
        partial(mse_mae_analysis, dataset.Y_test),
        rng,
        n_uncertain_samples=1,
        n_analysis_runs=10,
        progress=progress,
    )

    if progress is not None:
        progress.update_major(format="Calculating the Test Calibration Error")

    test_rmsce = calculate_calibration_error(
        dataset.Y_scaler.inverse_transform(dataset.Y_test),
        dataset.Y_scaler.inverse_transform(test_predictions.prediction),
        test_predictions.uncertainty * dataset.Y_scaler.scale_,
        test_predictions.confidence,
        progress=progress,
    )

    return TrainTestEvaluation(
        train_mse=IcarusPrediction(
            prediction=train_eval.prediction[0],
            uncertainty=train_eval.uncertainty[0],
            confidence=train_eval.confidence,
        ),
        train_mae=IcarusPrediction(
            prediction=train_eval.prediction[1],
            uncertainty=train_eval.uncertainty[1],
            confidence=train_eval.confidence,
        ),
        train_r2=IcarusPrediction(
            prediction=train_eval.prediction[2],
            uncertainty=train_eval.uncertainty[2],
            confidence=train_eval.confidence,
        ),
        train_rmsce=train_rmsce,
        test_mse=IcarusPrediction(
            prediction=test_eval.prediction[0],
            uncertainty=test_eval.uncertainty[0],
            confidence=test_eval.confidence,
        ),
        test_mae=IcarusPrediction(
            prediction=test_eval.prediction[1],
            uncertainty=test_eval.uncertainty[1],
            confidence=test_eval.confidence,
        ),
        test_r2=IcarusPrediction(
            prediction=test_eval.prediction[2],
            uncertainty=test_eval.uncertainty[2],
            confidence=test_eval.confidence,
        ),
        test_rmsce=test_rmsce,
    )


def generate_perturbed_predictions(
    model: IcarusRSM,
    dataset: MLDataset,
    rng,  #: np.random.Generator,
    n_samples: int,
    prediction_path: Path,
    overwrite_rsm_prediction: bool,
    perturbation,  #: Callable[[pd.DataFrame], pd.DataFrame],
    progress=None,
    **kwargs,
):  # -> pd.DataFrame:
    import joblib
    import numpy as np
    import pandas as pd

    if prediction_path.exists() and not overwrite_rsm_prediction:
        # FIXME: support saving to NetCDF files instead
        return joblib.load(prediction_path)

    if progress is not None:
        progress.update_major(
            value=0,
            min=0,
            max=3 + n_samples,
            format="Evaluating on the Test Dataset",
        )

    X_raw = perturbation(dataset.X_raw.copy(deep=True))

    if progress is not None:
        progress.update_major(format="Generating the Perturbed Dataset")

    X_prtb = dataset.X_scaler.transform(
        get_features_from_raw_features(X_raw, progress=progress)
    )
    Y_base = dataset.Y_raw

    prtb_predictions = []
    for i in range(n_samples):
        if progress is not None:
            progress.update_major(
                format=f"Predicting on the Perturbed Dataset {i}/{n_samples}"
            )

        prtb_predictions.append(model.predict(X_prtb, rng, progress=progress, **kwargs))

    if progress is not None:
        progress.update_major(
            format=f"Combining the Predictions on the Perturbed Dataset"
        )

        progress.update_minor(
            value=0,
            min=0,
            max=len(X_prtb),
            format="Perturbed Prediction %v/%m",
        )

    combined_prtb_predictions = IcarusPrediction(
        prediction=[],
        uncertainty=[],
        confidence=[],
    )

    for i in range(len(X_prtb)):
        predictions = np.array([p.prediction[i] for p in prtb_predictions])
        uncertainties = np.array([p.uncertainty[i] for p in prtb_predictions])
        confidences = np.array([p.confidence[i] for p in prtb_predictions])

        def combine_predictions(Y_pred, I_pred, rng, **kwargs):
            return (
                np.mean(Y_pred)
                if len(Y_pred) > 0
                else np.mean(
                    prtb_predictions[rng.choice(len(prtb_predictions))].prediction[i]
                )
            )

        cp = analyse_icarus_predictions(
            IcarusPrediction(
                prediction=predictions,
                uncertainty=uncertainties,
                confidence=confidences,
            ),
            combine_predictions,
            rng,
            n_uncertain_samples=1,
            n_analysis_runs=10,
            progress=None,
        )

        combined_prtb_predictions.prediction.append(cp.prediction)
        combined_prtb_predictions.uncertainty.append(cp.uncertainty)
        combined_prtb_predictions.confidence.append(cp.confidence)

        if progress is not None:
            progress.update_minor()

    prtb_predictions = IcarusPrediction(
        prediction=np.array(combined_prtb_predictions.prediction).reshape(-1, 1),
        uncertainty=np.array(combined_prtb_predictions.uncertainty).reshape(-1, 1),
        confidence=np.array(combined_prtb_predictions.confidence),
    )

    if progress is not None:
        progress.update_major(format="Assembling the Perturbation Prediction")
        progress.update_minor(value=0, format="")

    Y_pred = np.concatenate(
        [
            Y_base.index.get_level_values(0)
            .to_numpy()
            .reshape(
                (
                    Y_base.index.levels[0].size,
                    Y_base.index.levels[1].size,
                    1,
                )
            ),
            Y_base.index.get_level_values(1)
            .to_numpy()
            .reshape(
                (
                    Y_base.index.levels[0].size,
                    Y_base.index.levels[1].size,
                    1,
                )
            ),
            df_to_numpy(Y_base).reshape(
                (Y_base.index.levels[0].size, Y_base.index.levels[1].size, 1)
            ),
            dataset.Y_scaler.inverse_transform(prtb_predictions.prediction).reshape(
                (Y_base.index.levels[0].size, Y_base.index.levels[1].size, 1)
            ),
            (prtb_predictions.uncertainty * dataset.Y_scaler.scale_).reshape(
                (Y_base.index.levels[0].size, Y_base.index.levels[1].size, 1)
            ),
            prtb_predictions.confidence.reshape(
                (Y_base.index.levels[0].size, Y_base.index.levels[1].size, 1)
            ),
        ],
        axis=2,
    )

    df = pd.DataFrame(
        Y_pred.reshape(
            Y_pred.shape[0] * Y_pred.shape[1],
            Y_pred.shape[2],
        ),
        columns=[
            "time",
            "level",
            "log10_ccn_baseline",
            "log10_ccn_perturbed_pred",
            "log10_ccn_perturbed_stdv",
            "log10_ccn_perturbed_conf",
        ],
    ).set_index(["time", "level"])

    joblib.dump(df, prediction_path)

    return df
