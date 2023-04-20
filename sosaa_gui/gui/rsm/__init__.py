from pathlib import Path

from ..browse import browsePath
from ..syntax import PythonHighlighter
from .build import build_sosaa_rsm
from .plot import update_rsm_plots
from .predict import predict_sosaa_rsm
from .progress import RsmMajorMinorProgress


def init_rsm_gui(gui):
    gui.rsm_dataset = None
    gui.rsm_model = None
    gui.rsm_prediction = None
    gui.rsm_plots_dirty = True

    # Initialise the progress bar for RSM training and prediction
    gui.rsm_build_progress = RsmMajorMinorProgress(
        gui.rsm_build_progress_major,
        gui.rsm_build_progress_minor,
    )
    gui.rsm_build_progress.update_major(
        value=0, format="No SOSAA RSM is loaded"
    )
    gui.rsm_build_progress.update_minor(value=0, format="")

    gui.rsm_predict_progress = RsmMajorMinorProgress(
        gui.rsm_predict_progress_major,
        gui.rsm_predict_progress_minor,
    )
    gui.rsm_predict_progress.update_major(value=0, format="")
    gui.rsm_predict_progress.update_minor(value=0, format="")

    # Connect the RSM training and evaluation buttons
    gui.rsm_build.clicked.connect(
        lambda: build_sosaa_rsm(gui, rsm_should_exist=False)
    )
    gui.rsm_load.clicked.connect(
        lambda: build_sosaa_rsm(gui, rsm_should_exist=True)
    )

    # Connect and disable the RSM prediction button
    gui.rsm_predict.clicked.connect(lambda: predict_sosaa_rsm(gui))
    gui.rsm_predict.setEnabled(False)

    # Initialise the RSM configuration inputs
    gui.browse_rsm.clicked.connect(lambda: _change_rsm_file(gui))
    gui.browse_rsm_output.clicked.connect(
        lambda: _change_rsm_prediction_file(gui)
    )

    gui.rsm_perturbation.setPlaceholderText("return inputs")
    gui.rsm_perturbation_highlight = PythonHighlighter(
        gui,
        gui.rsm_perturbation.document(),
    )

    # Listen to RSM tab changes to lazily update the plots
    gui.rsm_subtab.currentChanged.connect(lambda i: _tab_switched(gui, i))


def _change_rsm_file(gui):
    path = browsePath(title="Choose the RSM file")

    if path is None:
        return

    if str(path).startswith(str(Path.cwd())):
        path = f"./{path.relative_to(Path.cwd())}"

    gui.rsm_path.setText(str(path))


def _change_rsm_prediction_file(gui):
    path = browsePath(title="Choose the RSM prediction output file")

    if path is None:
        return

    if str(path).startswith(str(Path.cwd())):
        path = f"./{path.relative_to(Path.cwd())}"

    gui.rsm_output.setText(str(path))


def _tab_switched(gui, _i):
    # Fake-access the i variable
    _i = _i

    if gui.rsm_subtab.currentWidget() == gui.rsm_prediction_tab:
        update_rsm_plots(gui)
