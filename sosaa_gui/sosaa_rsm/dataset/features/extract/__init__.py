from ...netcdf import TrajectoryDatasets
from .aerosol import get_aer_emissions_features
from .anthropogenic import get_ant_emissions_features
from .biogenic import get_bio_emissions_features
from .meteorology import get_meteorology_features


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
