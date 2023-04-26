import datetime
from collections import namedtuple
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


def get_sosaa_dataset_paths(
    dt: datetime.datetime, input_dir: Path, output_dir: Path
) -> TrajectoryPaths:
    out_path = output_dir / "output.nc"
    # fmt: on
    aer_path = (
        input_dir
        / f"{dt.strftime('%Y%m%d')}_7daybwd_Hyde_traj_AER_{24-dt.hour:02}_L3.nc"  # noqa: E501
    )
    ant_path = (
        input_dir
        / f"{dt.strftime('%Y%m%d')}_7daybwd_Hyde_traj_ANT_{24-dt.hour:02}_L3.nc"  # noqa: E501
    )
    bio_path = (
        input_dir
        / f"{dt.strftime('%Y%m%d')}_7daybwd_Hyde_traj_BIO_{24-dt.hour:02}_L3.nc"  # noqa: E501
    )
    # fmt: off
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
