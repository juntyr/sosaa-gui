from pathlib import Path

from PyQt5 import QtWidgets

from .worker import run_in_thread


def predict_sosaa_rsm(gui):
    # Disable concurrent training or prediction
    gui.rsm_build.setEnabled(False)
    gui.rsm_load.setEnabled(False)
    gui.rsm_predict.setEnabled(False)

    # Reset any previous RSM predictions
    gui.rsm_prediction = None

    # Generate the RSM predictions and update the GUI
    _generate_rsm_prediction_and_update_gui(gui)


def _generate_rsm_prediction_and_update_gui(gui):
    try:
        import numpy as np

        from ...sosaa_rsm.model.perturb import generate_perturbed_predictions

        # Configure the prediction
        n_samples = gui.rsm_predict_samples.value()
        prediction_path = Path(gui.rsm_output.text()).resolve()
        perturbation = _generate_perturbation_function(gui)

        predict_seed = np.random.SeedSequence(
            list(gui.rsm_predict_seed.text().encode())
        )
        predict_rng = np.random.RandomState(np.random.PCG64(predict_seed))

        if prediction_path.exists():
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.setStandardButtons(
                QtWidgets.QMessageBox.Open
                | QtWidgets.QMessageBox.Reset
                | QtWidgets.QMessageBox.Cancel
            )
            msg.setText(
                "Do you want to load the existing RSM prediction or"
                " overwrite it?"
            )
            msg.setInformativeText(
                f"The file {str(prediction_path)} already exists."
            )
            msg.setWindowTitle("Existing SOSAA RSM Prediction")
            button = msg.exec_()

            if button == QtWidgets.QMessageBox.Cancel:
                return _on_prediction_finished(gui, err=None, result=None)

            overwrite_rsm_prediction = button != QtWidgets.QMessageBox.Open
        else:
            overwrite_rsm_prediction = True

        # Generate the RSM predictions in a worker thread
        gui.rsm_predict_thread = run_in_thread(
            lambda: generate_perturbed_predictions(
                gui.rsm_model,
                gui.rsm_dataset,
                predict_rng,
                n_samples,
                prediction_path,
                overwrite_rsm_prediction,
                perturbation,
                gui.rsm_predict_progress,
            ),
            lambda result: _on_prediction_finished(
                gui, err=None, result=result
            ),
            lambda err: _on_prediction_finished(gui, err),
        )
    except Exception as err:
        _on_prediction_finished(gui, err)


def _on_prediction_finished(gui, err, result=None):
    try:
        if err is not None:
            raise err

        # Generate the perturbed prediction using the RSM
        gui.rsm_prediction = result

        # Early return if loading or predicting was cancelled
        if gui.rsm_prediction is None:
            return
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
        msg.setWindowTitle("Error predicting with SOSAA RSM")
        msg.exec_()

        return
    finally:
        gui.rsm_plots_dirty = True

        # Reset the RSM GUI to allow training a new RSM
        #  and making new predictions
        gui.rsm_predict_progress.update_major(value=0, format="")
        gui.rsm_predict_progress.update_minor(value=0, format="")

        gui.rsm_build.setEnabled(True)
        gui.rsm_load.setEnabled(True)
        gui.rsm_predict.setEnabled(True)

    # Communicate prediction success
    gui.rsm_predict_progress.update_major(
        value=1,
        max=1,
        format="Completed Predicting with the SOSAA RSM",
    )
    gui.rsm_predict_progress.update_minor(
        value=1,
        max=1,
        format=(
            f"The SOSAA RSM predictions are stored at {gui.rsm_output.text()}"
        ),
    )


def _generate_perturbation_function(gui):
    import pandas as pd

    perturbation_code = gui.rsm_perturbation.toPlainText()

    if len(perturbation_code) == 0:
        perturbation_code = "return inputs"

    # Source code for the perturbation 'sandbox'
    # The GUI specifies the body of the following function
    # def perturb_inputs(inputs: pandas.DataFrame) -> pandas.DataFrame:
    #     <BODY>
    perturbation_code = """import numpy
import pandas
import numpy as np
import pandas as pd

def perturb_inputs(inputs: pandas.DataFrame) -> pandas.DataFrame:
""" + "".join(f"    {line}" for line in perturbation_code.splitlines(True))

    def perturb_inputs_wrapper(inputs: pd.DataFrame) -> pd.DataFrame:
        locals = dict()
        globals = dict()

        exec(perturbation_code, globals, locals)

        return locals["perturb_inputs"](inputs)

    return perturb_inputs_wrapper
