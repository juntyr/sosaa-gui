from pathlib import Path

from ..dataset import MLDataset
from ..dataset.features import get_features_from_raw_features
from ..dataset.utils import df_to_numpy
from ..icarus import IcarusPrediction, IcarusRSM
from ..icarus.analyse import analyse_icarus_predictions


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

        prtb_predictions.append(
            model.predict(X_prtb, rng, progress=progress, **kwargs)
        )

    if progress is not None:
        progress.update_major(
            format="Combining the Predictions on the Perturbed Dataset"
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
                    prtb_predictions[
                        rng.choice(len(prtb_predictions))
                    ].prediction[i]
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
        prediction=np.array(combined_prtb_predictions.prediction).reshape(
            -1, 1
        ),
        uncertainty=np.array(combined_prtb_predictions.uncertainty).reshape(
            -1, 1
        ),
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
            dataset.Y_scaler.inverse_transform(
                prtb_predictions.prediction
            ).reshape(
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
