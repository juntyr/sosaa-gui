from PyQt5 import QtWidgets

from .ccn import init_plot as _init_ccn_plot, update_plot as _update_ccn_plot
from .diff import init_plot as _init_diff_plot, update_plot as _update_diff_plot


def init_rsm_plots(gui):
    if getattr(gui, "rsm_plot_init", False):
        return

    try:
        import matplotlib as mpl

        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
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

    NavigationToolbar.toolitems.pop(-3)
    NavigationToolbar.toolitems.pop(-3)

    _init_ccn_plot(gui)
    _init_diff_plot(gui)

    gui.rsm_plot_init = True


def update_rsm_plots(gui):
    init_rsm_plots(gui)

    _update_ccn_plot(gui)
    _update_diff_plot(gui)