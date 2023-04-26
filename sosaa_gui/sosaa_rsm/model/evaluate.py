from collections import namedtuple
from functools import partial

from ..dataset import MLDataset
from ..icarus import IcarusPrediction, IcarusRSM
from ..icarus.analyse import analyse_icarus_predictions

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
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

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
            format="Combining the Predictions on the Training Dataset"
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
                    train_predictions[
                        rng.choice(len(train_predictions))
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

        combined_train_predictions.prediction.append(cp.prediction)
        combined_train_predictions.uncertainty.append(cp.uncertainty)
        combined_train_predictions.confidence.append(cp.confidence)

        if progress is not None:
            progress.update_minor()

    train_predictions = IcarusPrediction(
        prediction=np.array(combined_train_predictions.prediction).reshape(
            -1, 1
        ),
        uncertainty=np.array(combined_train_predictions.uncertainty).reshape(
            -1, 1
        ),
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
        progress.update_major(
            format="Calculating the Training Calibration Error"
        )

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
        progress.update_major(
            format="Combining the Predictions on the Test Dataset"
        )

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
                    test_predictions[
                        rng.choice(len(test_predictions))
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

        combined_test_predictions.prediction.append(cp.prediction)
        combined_test_predictions.uncertainty.append(cp.uncertainty)
        combined_test_predictions.confidence.append(cp.confidence)

        if progress is not None:
            progress.update_minor()

    test_predictions = IcarusPrediction(
        prediction=np.array(combined_test_predictions.prediction).reshape(
            -1, 1
        ),
        uncertainty=np.array(combined_test_predictions.uncertainty).reshape(
            -1, 1
        ),
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
        progress.update_minor(
            value=0, min=0, max=N, format="Checking Percentile %p"
        )

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
