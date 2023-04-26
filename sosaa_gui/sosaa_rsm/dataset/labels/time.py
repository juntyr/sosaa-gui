import datetime

from ..netcdf import TrajectoryDatasets


def get_output_time(ds: TrajectoryDatasets):
    fdom = datetime.datetime.strptime(
        ds.out["time"].__dict__["first_day_of_month"],
        "%Y-%m-%d %H:%M:%S",
    )
    dt = (ds.date - fdom).total_seconds()

    out_t = ds.out["time"][:].data

    return out_t - dt
