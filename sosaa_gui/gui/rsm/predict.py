from pathlib import Path

from PyQt5 import QtWidgets

from .plot import update_rsm_plots


def predict_sosaa_rsm(gui):
    # Disable concurrent training or prediction
    gui.rsm_build.setEnabled(False)
    gui.rsm_load.setEnabled(False)
    gui.rsm_predict.setEnabled(False)

    # Reset any previous RSM predictions
    gui.rsm_prediction = None

    try:
        # Generate the perturbed prediction using the RSM
        gui.rsm_prediction = _generate_rsm_prediction(gui)

        # Early return if loading or predicting was cancelled
        if gui.rsm_prediction is None:
            return
    except ImportError as err:
        # Gracefully catch missing dependencies
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(f"Optional dependency {err.name} missing")
        msg.setInformativeText(
            "Please install sosaa-gui with the optional 'icarus' feature enabled."
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
        format=f"The SOSAA RSM predictions are stored at {gui.rsm_output.text()}",
    )

    # Update the result plots
    update_rsm_plots(gui)


def _generate_rsm_prediction(gui):
    import numpy as np

    from ...sosaa_rsm import generate_perturbed_predictions

    # Configure the prediction
    n_samples = gui.rsm_predict_samples.value()
    prediction_path = Path(gui.rsm_output.text()).resolve()
    perturbation = _generate_perturbation_function(gui)

    predict_seed = np.random.SeedSequence(list(gui.rsm_predict_seed.text().encode()))
    predict_rng = np.random.RandomState(np.random.PCG64(predict_seed))

    # Generate the RSM prediction
    return generate_perturbed_predictions(
        gui.rsm_model,
        gui.rsm_dataset,
        predict_rng,
        n_samples,
        prediction_path,
        perturbation,
        gui.rsm_predict_progress,
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
""" + "".join(
        f"    {line}" for line in perturbation_code.splitlines(True)
    )

    def perturb_inputs_wrapper(inputs: pd.DataFrame) -> pd.DataFrame:
        locals = dict()
        globals = dict()

        exec(perturbation_code, globals, locals)

        return locals["perturb_inputs"](inputs)

    return perturb_inputs_wrapper
