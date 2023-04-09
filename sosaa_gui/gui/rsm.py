import datetime

from pathlib import Path

from PyQt5 import QtWidgets

from .browse import browsePath
from .syntax import PythonHighlighter
from ..sosaa_rsm import (
    train_and_cache_model,
    RandomForestSosaaRSM,
    analyse_train_test_perforance,
    load_and_cache_dataset,
    IcarusPrediction,
)


def init_rsm_gui(gui):
    gui.rsm_dataset = None
    gui.rsm_model = None

    class RsmBuildProgress:
        def update_minor(self, value=None, min=None, max=None, format=None):
            if min is not None:
                gui.rsm_build_progress_low.setMinimum(min)

            if max is not None:
                gui.rsm_build_progress_low.setMaximum(max)

            if format is not None:
                gui.rsm_build_progress_low.setFormat(format)

            if value is not None:
                gui.rsm_build_progress_low.setValue(value)
            else:
                gui.rsm_build_progress_low.setValue(
                    gui.rsm_build_progress_low.value() + 1
                )

        def update_major(self, value=None, min=None, max=None, format=None):
            if min is not None:
                gui.rsm_build_progress_high.setMinimum(min)

            if max is not None:
                gui.rsm_build_progress_high.setMaximum(max)

            if format is not None:
                gui.rsm_build_progress_high.setFormat(format)

            if value is not None:
                gui.rsm_build_progress_high.setValue(value)
            else:
                gui.rsm_build_progress_high.setValue(
                    gui.rsm_build_progress_high.value() + 1
                )

    RsmBuildProgress().update_major(value=0, format="")
    RsmBuildProgress().update_minor(value=0, format="")

    def buildSosaaRsm():
        gui.rsm_train_mse.setText("")
        gui.rsm_train_mae.setText("")
        gui.rsm_train_r2.setText("")
        gui.rsm_test_mse.setText("")
        gui.rsm_test_mae.setText("")
        gui.rsm_test_r2.setText("")

        try:
            import numpy as np

            input_dir = Path(gui.input_dir.text()).resolve()
            output_dir = Path(gui.output_dir.text()).resolve()
            rsm_path = Path(gui.rsm_path.text()).resolve()
            dt = datetime.datetime(year=2018, month=5, day=15, hour=19)
            clump = 0.75  # sensible default
            datasets = dict()  # no caching
            models = dict()  # no caching
            n_trees = gui.rsm_forest.value()
            n_samples = gui.rsm_train_samples.value()
            progress = RsmBuildProgress()

            train_seed = np.random.SeedSequence(
                list(gui.rsm_train_seed.text().encode())
            )
            train_rng = np.random.RandomState(np.random.PCG64(train_seed))

            gui.rsm_dataset = None
            gui.rsm_model = None

            progress.update_major(
                value=0, min=0, max=6, format="Training the SOSAA RSM"
            )

            model = train_and_cache_model(
                dt,
                clump,
                datasets,
                models,
                RandomForestSosaaRSM,
                train_rng,
                input_dir,
                output_dir,
                rsm_path,
                n_trees=n_trees,
                progress=progress,
            )

            if model is None:
                return

            progress.update_major(format="Loading the SOSAA Dataset")

            dataset = load_and_cache_dataset(
                dt,
                clump,
                datasets,
                input_dir,
                output_dir,
                progress=progress,
            )

            eval_rng = np.random.RandomState(np.random.PCG64(train_seed))

            train_test_eval = analyse_train_test_perforance(
                model,
                dataset,
                eval_rng,
                progress=progress,
            )

            gui.rsm_dataset = dataset
            gui.rsm_model = model

            def fip(p: IcarusPrediction):
                return (
                    f"({p.prediction:.02} ± {p.uncertainty:.02}) | {p.confidence:.02}"
                )

            gui.rsm_train_mse.setText(fip(train_test_eval.train_mse))
            gui.rsm_train_mae.setText(fip(train_test_eval.train_mae))
            gui.rsm_train_r2.setText(fip(train_test_eval.train_r2))
            gui.rsm_test_mse.setText(fip(train_test_eval.test_mse))
            gui.rsm_test_mae.setText(fip(train_test_eval.test_mae))
            gui.rsm_test_r2.setText(fip(train_test_eval.test_r2))
        except ImportError as err:
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
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(type(err).__name__)
            msg.setInformativeText(str(err))
            msg.setWindowTitle("Error training SOSAA RSM")
            msg.exec_()

            return
        finally:
            RsmBuildProgress().update_major(value=0, format="")
            RsmBuildProgress().update_minor(value=0, format="")

        RsmBuildProgress().update_major(
            value=1, max=1, format="Completed training the SOSAA RSM"
        )

    gui.rsm_build.clicked.connect(buildSosaaRsm)

    def predictSosaaRsm():
        if gui.rsm_model is None:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("No SOSAA RSM is loaded in")
            msg.setWindowTitle("Error predicting with the SOSAA RSM")
            msg.exec_()

            return

        try:
            import numpy as np

            n_samples = gui.rsm_predict_samples.value()

            predict_seed = np.random.SeedSequence(
                list(gui.rsm_predict_seed.text().encode())
            )
            predict_rng = np.random.RandomState(np.random.PCG64(predict_seed))

        except ImportError as err:
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
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(type(err).__name__)
            msg.setInformativeText(str(err))
            msg.setWindowTitle("Error training SOSAA RSM")
            msg.exec_()

            return

    gui.rsm_predict.clicked.connect(predictSosaaRsm)
    gui.rsm_predict.setEnabled(False)

    def changeRsmFile():
        path = browsePath(title="Choose the RSM file")

        if path is None:
            return

        if str(path).startswith(str(Path.cwd())):
            path = f"./{path.relative_to(Path.cwd())}"

        gui.rsm_path.setText(str(path))

    gui.browse_rsm.clicked.connect(changeRsmFile)

    def changeRsmOutputFile():
        path = browsePath(title="Choose the RSM prediction output file")

        if path is None:
            return

        if str(path).startswith(str(Path.cwd())):
            path = f"./{path.relative_to(Path.cwd())}"

        gui.rsm_output.setText(str(path))

    gui.browse_rsm_output.clicked.connect(changeRsmOutputFile)

    gui.rsm_perturbation.setPlaceholderText("return inputs")
    gui.rsm_perturbation_highlight = PythonHighlighter(gui.rsm_perturbation.document())

    gui.rsm_subtab.currentChanged.connect(lambda i: _tab_switched(gui, i))


def _tab_switched(gui, _i):
    # Fake-access the i variable
    _i = _i

    if gui.rsm_subtab.currentWidget() == gui.rsm_prediction_tab:
        _lazy_init_plot(gui)


def _lazy_init_plot(gui):
    if getattr(gui, "rsm_plot_canvas", None) is not None:
        return

    try:
        import matplotlib as mpl

        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT as NavigationToolbar,
        )
    except ImportError as err:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(f"Optional dependency {err.name} missing")
        msg.setInformativeText(
            "Please install sosaa-gui with the optional 'icarus' feature enabled."
        )
        msg.setWindowTitle("Missing optional dependency")
        msg.exec_()

        return

    mpl.use("Qt5Agg")
    plt.style.use("seaborn-v0_8")

    gui.rsm_plot_fig, gui.rsm_plot_ax = plt.subplots(1, 1, figsize=(6, 4), dpi=100)
    gui.rsm_plot_canvas = FigureCanvasQTAgg(gui.rsm_plot_fig)

    gui.rsm_plot_ax.set_title("Change in Cloud-Condensation-Nuclei (CCN) Concentration")
    gui.rsm_plot_ax.set_xlabel("Baseline CCN concentration")
    gui.rsm_plot_ax.set_xticks([])
    gui.rsm_plot_ax.set_ylabel("Perturbed Change in CCN concentration")
    gui.rsm_plot_ax.set_yticks([])
    gui.rsm_plot_fig.tight_layout()

    gui.rsm_plot_note = gui.rsm_plot_ax.text(
        0.5,
        0.5,
        "Missing SOSAA RSM Predictions",
        color="red",
        size=20,
        ha="center",
        va="center",
        transform=gui.rsm_plot_ax.transAxes,
    )

    NavigationToolbar.toolitems.pop(-3)
    NavigationToolbar.toolitems.pop(-3)
    gui.rsm_toolbar = NavigationToolbar(gui.rsm_plot_canvas, gui)

    predict_layout = QtWidgets.QVBoxLayout()
    predict_layout.addWidget(gui.rsm_toolbar)
    predict_layout.addWidget(gui.rsm_plot_canvas)

    gui.rsm_prediction_tab.setLayout(predict_layout)
