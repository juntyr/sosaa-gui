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
    generate_perturbed_predictions,
)


def init_rsm_gui(gui):
    gui.rsm_dataset = None
    gui.rsm_model = None
    gui.rsm_prediction = None

    class RsmProgress:
        def __init__(self, progress):
            self.progress = progress

        def update(self, value=None, min=None, max=None, format=None):
            if min is not None:
                self.progress.setMinimum(min)

            if max is not None:
                self.progress.setMaximum(max)

            if format is not None:
                self.progress.setFormat(format)

            if value is not None:
                self.progress.setValue(value)
            else:
                self.progress.setValue(self.progress.value() + 1)

    class RsmBuildProgress:
        def __init__(self):
            self.minor = RsmProgress(gui.rsm_build_progress_minor)
            self.major = RsmProgress(gui.rsm_build_progress_major)

        def update_minor(self, *args, **kwargs):
            self.minor.update(*args, **kwargs)

        def update_major(self, *args, **kwargs):
            self.major.update(*args, **kwargs)

    class RsmPredictProgress:
        def __init__(self):
            self.minor = RsmProgress(gui.rsm_predict_progress_minor)
            self.major = RsmProgress(gui.rsm_predict_progress_major)

        def update_minor(self, *args, **kwargs):
            self.minor.update(*args, **kwargs)

        def update_major(self, *args, **kwargs):
            self.major.update(*args, **kwargs)

    RsmBuildProgress().update_major(value=0, format="No SOSAA RSM is loaded")
    RsmBuildProgress().update_minor(value=0, format="")

    RsmPredictProgress().update_major(value=0, format="")
    RsmPredictProgress().update_minor(value=0, format="")

    def buildSosaaRsm(rsm_should_exist: bool):
        gui.rsm_build.setEnabled(False)
        gui.rsm_load.setEnabled(False)
        gui.rsm_predict.setEnabled(False)

        gui.rsm_train_mse.setText("")
        gui.rsm_train_mae.setText("")
        gui.rsm_train_r2.setText("")
        gui.rsm_train_rmsce.setText("")
        gui.rsm_test_mse.setText("")
        gui.rsm_test_mae.setText("")
        gui.rsm_test_r2.setText("")
        gui.rsm_test_rmsce.setText("")

        try:
            import numpy as np

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
            gui.rsm_prediction = None

            progress.update_major(
                value=0, min=0, max=6 + 2 * n_samples, format="Training the SOSAA RSM"
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
                rsm_should_exist,
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
                n_samples,
                progress=progress,
            )

            gui.rsm_dataset = dataset
            gui.rsm_model = model

            def fip(p: IcarusPrediction):
                if p.uncertainty is not None:
                    return f"({p.prediction:.02} ± {p.uncertainty:.02}) | {p.confidence:.02}"
                else:
                    return f"{p.prediction:.02} | {p.confidence:.02}"

            gui.rsm_train_mse.setText(fip(train_test_eval.train_mse))
            gui.rsm_train_mae.setText(fip(train_test_eval.train_mae))
            gui.rsm_train_r2.setText(fip(train_test_eval.train_r2))
            gui.rsm_train_rmsce.setText(fip(train_test_eval.train_rmsce))
            gui.rsm_test_mse.setText(fip(train_test_eval.test_mse))
            gui.rsm_test_mae.setText(fip(train_test_eval.test_mae))
            gui.rsm_test_r2.setText(fip(train_test_eval.test_r2))
            gui.rsm_test_rmsce.setText(fip(train_test_eval.test_rmsce))
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
            RsmBuildProgress().update_major(value=0, format="No SOSAA RSM is loaded")
            RsmBuildProgress().update_minor(value=0, format="")

            gui.rsm_build.setEnabled(True)
            gui.rsm_load.setEnabled(True)

        gui.rsm_predict.setEnabled(True)

        RsmBuildProgress().update_major(
            value=1,
            max=1,
            format="Completed {} the SOSAA RSM".format(
                "loading" if rsm_should_exist else "training"
            ),
        )
        RsmBuildProgress().update_minor(
            value=1, max=1, format=f"The SOSAA RSM is stored at {str(rsm_path)}"
        )

    # FIXME: move to a worker thread
    gui.rsm_build.clicked.connect(lambda: buildSosaaRsm(rsm_should_exist=False))
    gui.rsm_load.clicked.connect(lambda: buildSosaaRsm(rsm_should_exist=True))

    def predictSosaaRsm():
        gui.rsm_build.setEnabled(False)
        gui.rsm_load.setEnabled(False)
        gui.rsm_predict.setEnabled(False)

        try:
            import numpy as np
            import pandas as pd

            n_samples = gui.rsm_predict_samples.value()
            prediction_path = Path(gui.rsm_output.text()).resolve()
            perturbation_code = gui.rsm_perturbation.toPlainText()

            if len(perturbation_code) == 0:
                perturbation_code = "return inputs"

            perturbation_code = """import numpy
import pandas
import numpy as np
import pandas as pd

def perturb_inputs(inputs: pandas.DataFrame) -> pandas.DataFrame:
""" + "".join(
                f"    {line}" for line in perturbation_code.splitlines(True)
            )

            predict_seed = np.random.SeedSequence(
                list(gui.rsm_predict_seed.text().encode())
            )
            predict_rng = np.random.RandomState(np.random.PCG64(predict_seed))
            progress = RsmPredictProgress()

            gui.rsm_prediction = None

            def perturb_inputs_wrapper(inputs: pd.DataFrame) -> pd.DataFrame:
                locals = dict()
                globals = dict()

                exec(perturbation_code, globals, locals)

                return locals["perturb_inputs"](inputs)

            perturbed_prediction = generate_perturbed_predictions(
                gui.rsm_model,
                gui.rsm_dataset,
                predict_rng,
                n_samples,
                prediction_path,
                perturb_inputs_wrapper,
                progress,
            )

            if perturbed_prediction is None:
                return

            gui.rsm_prediction = perturbed_prediction
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
            RsmPredictProgress().update_major(value=0, format="")
            RsmPredictProgress().update_minor(value=0, format="")

            gui.rsm_build.setEnabled(True)
            gui.rsm_load.setEnabled(True)
            gui.rsm_predict.setEnabled(True)

        RsmPredictProgress().update_major(
            value=1,
            max=1,
            format="Completed Predicting with the SOSAA RSM",
        )
        RsmPredictProgress().update_minor(
            value=1,
            max=1,
            format=f"The SOSAA RSM predictions are stored at {str(prediction_path)}",
        )

        _lazy_init_plot(gui)
        _plot_rsm_prediction(gui)

    # FIXME: move to a worker thread
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

        if gui.rsm_prediction is None:
            _plot_rsm_prediction(gui)


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

    gui.rsm_plot_cb = gui.rsm_plot_fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=None,
            cmap="viridis",
        ),
        ax=gui.rsm_plot_ax,
        orientation="horizontal",
        extend="min",
    )

    NavigationToolbar.toolitems.pop(-3)
    NavigationToolbar.toolitems.pop(-3)
    gui.rsm_toolbar = NavigationToolbar(gui.rsm_plot_canvas, gui)

    predict_layout = QtWidgets.QVBoxLayout()
    predict_layout.addWidget(gui.rsm_toolbar)
    predict_layout.addWidget(gui.rsm_plot_canvas)

    gui.rsm_prediction_tab.setLayout(predict_layout)


def _plot_rsm_prediction(gui):
    import matplotlib as mpl
    import numpy as np

    gui.rsm_plot_ax.cla()

    gui.rsm_plot_ax.set_title("Change in Cloud-Condensation-Nuclei (CCN) Concentration")

    gui.rsm_plot_ax.set_xlabel("Baseline CCN concentration [m$^{-3}$]")
    gui.rsm_plot_ax.set_ylabel("Perturbed Change in CCN concentration [m$^{-3}$]")

    if gui.rsm_prediction is None:
        gui.rsm_plot_ax.set_xscale("linear")
        gui.rsm_plot_ax.set_yscale("linear")

        gui.rsm_plot_ax.set_xticks([])
        gui.rsm_plot_ax.set_yticks([])

        gui.rsm_plot_cb.ax.set_xticks([])

        gui.rsm_plot_ax.text(
            0.5,
            0.5,
            "Missing SOSAA RSM Predictions",
            color="red",
            size=20,
            ha="center",
            va="center",
            transform=gui.rsm_plot_ax.transAxes,
        )

        gui.rsm_plot_fig.tight_layout()
        gui.rsm_plot_canvas.draw()

        return

    time = gui.rsm_prediction.index.get_level_values(0)
    log10_ccn_baseline = gui.rsm_prediction["log10_ccn_baseline"].to_numpy().flatten()
    log10_ccn_perturbed_pred = (
        gui.rsm_prediction["log10_ccn_perturbed_pred"].to_numpy().flatten()
    )
    log10_ccn_perturbed_stdv = (
        gui.rsm_prediction["log10_ccn_perturbed_stdv"].to_numpy().flatten()
    )
    log10_ccn_perturbed_conf = (
        gui.rsm_prediction["log10_ccn_perturbed_conf"].to_numpy().flatten()
    )

    gui.rsm_plot_ax.set_title(
        "Change in Cloud-Condensation-Nuclei (CCN) Concentration"
        + f"\nConfidence: {np.mean(log10_ccn_perturbed_conf):.02}"
    )

    gui.rsm_plot_ax.set_xscale("log")
    gui.rsm_plot_ax.set_yscale(
        "symlog", linthresh=10 ** int(np.amin(log10_ccn_baseline))
    )

    for _ in range(gui.rsm_train_samples.value()):
        log10_ccn_perturbed = np.random.normal(
            loc=log10_ccn_perturbed_pred,
            scale=log10_ccn_perturbed_stdv,
        )

        gui.rsm_plot_ax.scatter(
            np.power(10.0, log10_ccn_baseline),
            np.power(10.0, log10_ccn_perturbed) - np.power(10.0, log10_ccn_baseline),
            alpha=log10_ccn_perturbed_conf,
            c=time,
            cmap="viridis",
            s=5,
        )

    xlim = gui.rsm_plot_ax.get_xlim()
    gui.rsm_plot_ax.plot(xlim, [0, 0], c="black", lw=1)
    gui.rsm_plot_ax.set_xlim(xlim)

    gui.rsm_plot_cb.ax.set_xticks(
        [
            (
                (h - int((gui.rsm_prediction.index.levels[0][0] // (60 * 60))))
                / (-int((gui.rsm_prediction.index.levels[0][0] // (60 * 60))))
            )
            for h in range(
                int((gui.rsm_prediction.index.levels[0][0] // (60 * 60))), 0, 24
            )
        ]
    )
    gui.rsm_plot_cb.ax.set_xticklabels(
        [
            (gui.rsm_dt + datetime.timedelta(hours=h)).strftime("%d.%m")
            for h in range(
                int((gui.rsm_prediction.index.levels[0][0] // (60 * 60))), 0, 24
            )
        ]
    )

    gui.rsm_plot_fig.tight_layout()
    gui.rsm_plot_canvas.draw()
