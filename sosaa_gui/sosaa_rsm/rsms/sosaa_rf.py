from __future__ import annotations

from ..icarus import IcarusPrediction, IcarusRSM


# SOSAA-RF with a percentile-based confidence score
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
        **kwargs,
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
                value=0,
                min=0,
                max=5,
                format="Fitting the Truncated PCA OOD Detector",
            )

        self.pca = PCA(random_state=rng).fit(X_train)
        self.bn = np.searchsorted(
            np.cumsum(self.pca.explained_variance_ratio_), 0.95
        )

        if progress is not None:
            progress.update_minor(
                format="Fitting the Auto-Associative Error Covariance"
            )

        self.cov = EmpiricalCovariance().fit(
            (self._predict_truncated_pca(X_train) - X_train)
        )

        self.err_valid = np.sort(
            self.cov.mahalanobis(
                self._predict_truncated_pca(X_valid) - X_valid
            )
        )

        if progress is not None:
            progress.update_minor(
                format=(
                    "Training the Prediction Model and Uncertainty Quantifier"
                )
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
                format="Generating the Confidence Scores",
            )

        confidence = 1.0 - np.searchsorted(
            self.err_valid,
            self.cov.mahalanobis(
                (self._predict_truncated_pca(X_test) - X_test)
            ),
        ) / len(self.err_valid)

        if progress is not None:
            progress.update_minor(
                format=(
                    "Generating"
                    f" {0}/{len(self.predictor.estimators_)} Ensemble"
                    " Predictions"
                )
            )

        def tree_predict(i, tree, X_test, progress=None) -> np.ndarray:
            result = tree.predict(X_test)

            if progress is not None:
                progress.update_minor(
                    format=(
                        "Generating"
                        f" {i}/{len(self.predictor.estimators_)} Ensemble"
                        " Predictions"
                    )
                )

            return result

        predictions = joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(tree_predict)(i, tree, X_test, progress)
            for i, tree in enumerate(self.predictor.estimators_)
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
