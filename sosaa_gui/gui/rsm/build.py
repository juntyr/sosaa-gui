import datetime
from pathlib import Path

from PyQt5 import QtWidgets

from .worker import run_in_thread


def build_sosaa_rsm(gui, rsm_should_exist: bool):
    # Disable concurrent training or prediction
    gui.rsm_build.setEnabled(False)
    gui.rsm_load.setEnabled(False)
    gui.rsm_predict.setEnabled(False)

    # Reset the RSM evaluation metrics
    gui.rsm_train_mse.setText("")
    gui.rsm_train_mae.setText("")
    gui.rsm_train_r2.setText("")
    gui.rsm_train_rmsce.setText("")
    gui.rsm_test_mse.setText("")
    gui.rsm_test_mae.setText("")
    gui.rsm_test_r2.setText("")
    gui.rsm_test_rmsce.setText("")

    # Reset the RSM training results
    gui.rsm_dataset = None
    gui.rsm_model = None
    gui.rsm_prediction = None

    # Build the RSM, evaluate it, and update the GUI
    _train_evaluate_sosaa_rsm(gui, rsm_should_exist)


def _train_evaluate_sosaa_rsm(gui, rsm_should_exist: bool):
    try:
        import numpy as np

        # SOSAA-RF has slightly more accurate predictions,
        #  has better performance, and works well with the
        #  percentile-based OOD scorer.
        # Future work can take the thesis evaluation
        #  experiment as inspiration for how to specialise
        #  support for SOSAA-PADRE-RF (e.g. with direct
        #  perturbation predictions)
        # from ...sosaa_rsm.rsms.sosaa_padre_rf import (
        #     PairwiseDifferenceRegressionRandomForestSosaaRSM as SosaaRSM,
        # )
        from ...sosaa_rsm.rsms.sosaa_rf import RandomForestSosaaRSM as SosaaRSM

        # Configure the RSM
        input_dir = Path(gui.input_dir.text()).resolve()
        output_dir = Path(gui.output_dir.text()).resolve()
        rsm_path = Path(gui.rsm_path.text()).resolve()
        dt = gui.end_date.dateTime()
        dt = datetime.datetime(
            year=dt.date().year(),
            month=dt.date().month(),
            day=dt.date().day(),
            hour=dt.time().hour(),
            minute=dt.time().minute(),
            second=dt.time().second(),
        )
        gui.rsm_dt = dt
        clump = 0.75  # sensible default
        n_trees = gui.rsm_forest.value()
        n_samples = gui.rsm_train_samples.value()

        train_seed = np.random.SeedSequence(
            list(gui.rsm_train_seed.text().encode())
        )
        train_rng = np.random.RandomState(np.random.PCG64(train_seed))
        eval_rng = np.random.RandomState(np.random.PCG64(train_seed))

        overwrite_rsm = False

        if rsm_path.exists():
            if not rsm_should_exist:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Question)
                msg.setStandardButtons(
                    QtWidgets.QMessageBox.Open
                    | QtWidgets.QMessageBox.Reset
                    | QtWidgets.QMessageBox.Cancel
                )
                msg.setText(
                    "Do you want to load the existing RSM or overwrite it?"
                )
                msg.setInformativeText(
                    f"The file {str(rsm_path)} already exists."
                )
                msg.setWindowTitle("Existing SOSAA RSM")
                button = msg.exec_()

                if button == QtWidgets.QMessageBox.Cancel:
                    return _on_build_finished(
                        gui, rsm_should_exist, err=None, result=None
                    )

                overwrite_rsm = button != QtWidgets.QMessageBox.Open
        elif rsm_should_exist:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.setStandardButtons(
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel
            )
            msg.setText("Do you want to train a new SOSAA RSM?")
            msg.setInformativeText(f"The file {str(rsm_path)} does not exist.")
            msg.setWindowTitle("Missing SOSAA RSM")
            button = msg.exec_()

            if button == QtWidgets.QMessageBox.Cancel:
                return _on_build_finished(
                    gui, rsm_should_exist, err=None, result=None
                )

        # Build and evaluate the RSM in a worker thread
        gui.rsm_build_thread = run_in_thread(
            lambda: _train_evaluate_sosaa_rsm_job(
                gui,
                overwrite_rsm,
                input_dir,
                output_dir,
                rsm_path,
                dt,
                clump,
                SosaaRSM,
                n_trees,
                n_samples,
                train_rng,
                eval_rng,
            ),
            lambda result: _on_build_finished(
                gui, rsm_should_exist, err=None, result=result
            ),
            lambda err: _on_build_finished(gui, rsm_should_exist, err),
        )
    except Exception as err:
        _on_build_finished(gui, rsm_should_exist, err)


def _train_evaluate_sosaa_rsm_job(
    gui,
    overwrite_rsm,
    input_dir,
    output_dir,
    rsm_path,
    dt,
    clump,
    cls,
    n_trees,
    n_samples,
    train_rng,
    eval_rng,
):
    from ...sosaa_rsm.dataset import load_and_cache_dataset
    from ...sosaa_rsm.model import train_and_cache_model
    from ...sosaa_rsm.model.evaluate import analyse_train_test_perforance

    gui.rsm_build_progress.update_major(
        value=0, min=0, max=6 + 2 * n_samples, format="Training the SOSAA RSM"
    )

    # Datasets and models are not cached between runs
    datasets = dict()
    models = dict()

    # Train the RSM model
    model = train_and_cache_model(
        dt,
        clump,
        datasets,
        models,
        cls,
        train_rng,
        input_dir,
        output_dir,
        rsm_path,
        overwrite_rsm,
        n_trees=n_trees,
        n_samples=n_samples,
        progress=gui.rsm_build_progress,
    )

    # Early return if loading or training was cancelled
    if model is None:
        return

    gui.rsm_build_progress.update_major(format="Loading the SOSAA Dataset")

    # Load the dataset again
    #  only needed if the model was loaded from disk
    dataset = load_and_cache_dataset(
        dt,
        clump,
        datasets,
        input_dir,
        output_dir,
        progress=gui.rsm_build_progress,
    )

    # Evaluate the model's train/test performance
    train_test_eval = analyse_train_test_perforance(
        model,
        dataset,
        eval_rng,
        n_samples,
        progress=gui.rsm_build_progress,
    )

    return (model, dataset, train_test_eval)


def _on_build_finished(gui, rsm_should_exist: bool, err, result=None):
    try:
        from ...sosaa_rsm.icarus import IcarusPrediction

        if err is not None:
            raise err

        # Early return if loading or training was cancelled
        if result is None:
            return

        model, dataset, train_test_eval = result
        gui.rsm_dataset = dataset
        gui.rsm_model = model

        # Pretty-print an Icarus prediction
        def fip(p: IcarusPrediction):
            if p.uncertainty is not None:
                return (
                    f"({p.prediction:.02} ± {p.uncertainty:.02}) |"
                    f" {p.confidence:.02}"
                )
            else:
                return f"{p.prediction:.02} | {p.confidence:.02}"

        # Output the new model evaluation results
        gui.rsm_train_mse.setText(fip(train_test_eval.train_mse))
        gui.rsm_train_mae.setText(fip(train_test_eval.train_mae))
        gui.rsm_train_r2.setText(fip(train_test_eval.train_r2))
        gui.rsm_train_rmsce.setText(fip(train_test_eval.train_rmsce))
        gui.rsm_test_mse.setText(fip(train_test_eval.test_mse))
        gui.rsm_test_mae.setText(fip(train_test_eval.test_mae))
        gui.rsm_test_r2.setText(fip(train_test_eval.test_r2))
        gui.rsm_test_rmsce.setText(fip(train_test_eval.test_rmsce))
    except ImportError as err:
        # Gracefully catch missing dependencies
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(f"Optional dependency {err.name} missing")
        msg.setInformativeText(
            "Please install sosaa-gui with the optional 'icarus' feature"
            " enabled."
        )
        msg.setWindowTitle("Missing optional dependency")
        msg.exec_()

        return
    except Exception as err:
        # Gracefully catch internal errors
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(type(err).__name__)
        msg.setInformativeText(str(err))
        msg.setWindowTitle("Error training SOSAA RSM")
        msg.exec_()

        return
    finally:
        gui.rsm_plots_dirty = True

        # Reset the RSM GUI to allow training a new RSM
        gui.rsm_build_progress.update_major(
            value=0, format="No SOSAA RSM is loaded"
        )
        gui.rsm_build_progress.update_minor(value=0, format="")

        gui.rsm_build.setEnabled(True)
        gui.rsm_load.setEnabled(True)

    # Communicate training success and allow RSM predictions
    gui.rsm_predict.setEnabled(True)

    gui.rsm_build_progress.update_major(
        value=1,
        max=1,
        format="Completed {} the SOSAA RSM".format(
            "loading" if rsm_should_exist else "training"
        ),
    )
    gui.rsm_build_progress.update_minor(
        value=1,
        max=1,
        format=f"The SOSAA RSM is stored at {gui.rsm_path.text()}",
    )
