from pathlib import Path

import matplotlib as mpl

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)

from PyQt5 import QtWidgets

from .browse import browsePath
from .syntax import PythonHighlighter


def init_rsm_gui(gui):
    gui.rsm_build.setEnabled(False)
    gui.rsm_load.setEnabled(False)
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
