from collections import namedtuple

from .paths import TrajectoryPaths

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
