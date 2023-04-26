from . import IcarusPrediction


def analyse_icarus_predictions(
    predictions: IcarusPrediction,
    analysis,  #: Callable[[np.ndarray, np.ndarray,
    #                       np.random.Generator, dict,
    #                      ], np.ndarray],
    rng,  #: np.random.Generator,
    # number of samples to draw from expand each prediction per run
    n_uncertain_samples: int = 1,
    # number of repeats of the analysis to gather uncertainty
    n_analysis_runs: int = 100,
    **kwargs,
):
    import numpy as np

    progress = kwargs.get("progress", None)

    if progress is not None:
        progress.update_minor(
            value=0,
            min=0,
            max=n_analysis_runs,
            format="Monte Carlo Analysis Run %v/%m",
        )

    confidence = np.mean(predictions.confidence)

    results = []

    for _ in range(n_analysis_runs):
        confs = []
        preds = []
        for _ in range(n_uncertain_samples):
            I_conf = (
                rng.random(size=predictions.confidence.shape)
                <= predictions.confidence
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
