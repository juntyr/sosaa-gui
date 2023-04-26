from __future__ import annotations

from ..icarus import IcarusPrediction, IcarusRSM


# SOSAA-PADRE-RF with a logistic-regression-based confidence score
class PairwiseDifferenceRegressionRandomForestSosaaRSM(IcarusRSM):
    def fit(
        self,
        X_train,  # : np.ndarray,
        Y_train,  # : np.ndarray,
        X_valid,  # : np.ndarray,
        Y_valid,  # : np.ndarray,
        rng,  # : np.random.Generator,
        n_trees: int = 16,
        n_samples: int = 16,
        progress=None,
        **kwargs,
    ) -> PairwiseDifferenceRegressionRandomForestSosaaRSM:
        import joblib
        import numpy as np
        import sklearn
        from sklearn.covariance import EmpiricalCovariance
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        assert Y_train.shape[1:] == (1,)

        self.X_train = X_train
        self.Y_train = Y_train

        if progress is not None:
            progress.update_minor(
                value=0,
                min=0,
                max=7 + n_trees,
                format="Resampling the training and validation datasets",
            )

        Ia_train = rng.choice(
            len(self.X_train), size=len(X_train) * n_samples, replace=True
        )
        Ib_train = rng.choice(
            len(X_train), size=len(X_train) * n_samples, replace=True
        )
        Ia_valid = rng.choice(
            len(self.X_train), size=len(X_valid) * n_samples, replace=True
        )
        Ib_valid = rng.choice(
            len(X_valid), size=len(X_valid) * n_samples, replace=True
        )

        # N(0,1)-N(0,1) ~ N(0,2) -> divide by sqrt(2)
        #  s.t. all features are N(0,1)
        X_train = np.concatenate(
            [
                self.X_train[Ia_train],
                (X_train[Ib_train] - self.X_train[Ia_train]) / np.sqrt(2.0),
            ],
            axis=1,
        )
        Y_train = Y_train[Ib_train] - self.Y_train[Ia_train]

        X_valid = np.concatenate(
            [
                self.X_train[Ia_valid],
                (X_valid[Ib_valid] - self.X_train[Ia_valid]) / np.sqrt(2.0),
            ],
            axis=1,
        )
        Y_valid = Y_valid[Ib_valid] - self.Y_train[Ia_valid]

        if progress is not None:
            progress.update_minor(
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

        if progress is not None:
            progress.update_minor(
                format="Synthesizing OOD inputs with the FGSM"
            )

        adv_grad = self.pca.components_[self.bn]

        X_ood = rng.normal(loc=X_valid, scale=0.01) + np.sign(
            adv_grad
        ) * np.abs(
            rng.normal(loc=2.0, scale=0.5, size=(len(X_valid), 1))
        ) * rng.choice(
            [-1, 1]
        )

        if progress is not None:
            progress.update_minor(
                format="Training the OOD logistic regression classifier"
            )

        M_id = self.cov.mahalanobis(
            self._predict_truncated_pca(X_valid) - X_valid
        )
        M_ood = self.cov.mahalanobis(
            self._predict_truncated_pca(X_ood) - X_ood
        )

        self.scaler = StandardScaler().fit(M_id.reshape(-1, 1))

        self.ood_detector = LogisticRegression(
            penalty=None if sklearn.__version__ >= "1.2" else "none",
            class_weight="balanced",
            random_state=rng,
        ).fit(
            np.concatenate(
                [
                    self.scaler.transform(M_id.reshape(-1, 1)),
                    self.scaler.transform(M_ood.reshape(-1, 1)),
                ],
                axis=0,
            ).reshape(-1, 1),
            np.concatenate(
                [
                    np.ones(len(M_id)),
                    np.zeros(len(M_ood)),
                ],
                axis=0,
            ),
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
            progress.update_minor(
                format=f"Calibrating the Uncertainty Quantifier {0}/{n_trees}"
            )

        def tree_predict(i: int) -> np.ndarray:
            result = self.predictor.estimators_[i].predict(X_valid)

            if progress is not None:
                progress.update_minor(
                    format=(
                        f"Calibrating the Uncertainty Quantifier {i}/{n_trees}"
                    )
                )

            return result

        valid_predictions = joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(tree_predict)(i)
            for i in range(len(self.predictor.estimators_))
        )

        Y_valid_pred = np.mean(np.stack(valid_predictions, axis=0), axis=0)
        Y_valid_stdv = np.std(np.stack(valid_predictions, axis=0), axis=0)

        Zc = (
            Y_valid.flatten() - Y_valid_pred.flatten()
        ) / Y_valid_stdv.flatten()

        self.Zc_mean = np.mean(Zc)
        self.Zc_stdv = np.std(Zc)

        if progress is not None:
            progress.update_minor(format="Finished Training the SOSAA RSM")

        return self

    def predict(
        self,
        X_test,  # : np.ndarray,
        rng,  # : np.random.Generator,
        direct_difference: bool = False,
        X_base=None,  # : np.ndarray = None,
        Y_base=None,  # : np.ndarray = None,
        progress=None,
    ) -> IcarusPrediction:
        import joblib
        import numpy as np

        if progress is not None:
            progress.update_minor(
                value=0,
                min=0,
                max=(3 + len(self.predictor.estimators_)),
                format="Resampling the input dataset",
            )

        if not direct_difference:
            # Only one anchor sample is produced, call predict several
            #  times for more
            Ia_test = rng.choice(
                len(self.X_train), size=len(X_test), replace=True
            )
            X_train = self.X_train
            Y_train = self.Y_train
        else:
            # Direct difference mode can be used to directly predict
            #  perturbations if the true output for the baseline is known
            if (X_base is not None) and (Y_base is not None):
                X_train = X_base
                Y_train = Y_base
            else:
                X_train = self.X_train
                Y_train = self.Y_train

            assert len(X_test) == len(X_train)
            assert len(X_train) == len(Y_train)

            Ia_test = np.arange(len(X_train))

        # N(0,1)-N(0,1) ~ N(0,2) -> divide by sqrt(2)
        #  s.t. all features are N(0,1)
        X_test = np.concatenate(
            [
                X_train[Ia_test],
                (X_test - X_train[Ia_test]) / np.sqrt(2.0),
            ],
            axis=1,
        )

        if progress is not None:
            progress.update_minor(
                format="Generating the Confidence Scores",
            )

        confidence = self.ood_detector.predict_proba(
            self.scaler.transform(
                self.cov.mahalanobis(
                    self._predict_truncated_pca(X_test) - X_test
                ).reshape(-1, 1)
            )
        )[:, 1]

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

        if progress is not None:
            progress.update_minor(
                format="Recalibrating predictions and uncertainties",
            )

        prediction = Y_train[Ia_test] + (
            prediction.flatten() + self.Zc_mean * uncertainty.flatten()
        ).reshape((len(X_test), 1))
        uncertainty = (uncertainty.flatten() * self.Zc_stdv).reshape(
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
