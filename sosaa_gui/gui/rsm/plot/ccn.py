import datetime

from PyQt5 import QtWidgets


def init_plot(gui):
    if getattr(gui, "rsm_ccnplot_canvas", None) is not None:
        return

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )

    gui.rsm_ccnplot_fig, gui.rsm_ccnplot_ax = plt.subplots(
        1, 1, figsize=(6, 4), dpi=100
    )
    gui.rsm_ccnplot_canvas = FigureCanvasQTAgg(gui.rsm_ccnplot_fig)

    gui.rsm_ccnplot_cb = gui.rsm_ccnplot_fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=None,
            cmap="rainbow",
        ),
        ax=gui.rsm_ccnplot_ax,
        orientation="vertical",
    )

    ccnplot_layout = QtWidgets.QVBoxLayout()
    ccnplot_layout.addWidget(NavigationToolbar(gui.rsm_ccnplot_canvas, gui))
    ccnplot_layout.addWidget(gui.rsm_ccnplot_canvas)

    gui.rsm_ccnplot_tab.setLayout(ccnplot_layout)


def update_plot(gui):
    import matplotlib as mpl
    import numpy as np

    gui.rsm_ccnplot_ax.cla()

    gui.rsm_ccnplot_ax.set_title("Predicted CCN Concentration Profile")

    gui.rsm_ccnplot_ax.set_xlabel("Trajectory Timeline")
    gui.rsm_ccnplot_ax.set_ylabel("CCN concentration [m$^{-3}$]")

    if gui.rsm_prediction is None:
        gui.rsm_ccnplot_ax.set_xscale("linear")
        gui.rsm_ccnplot_ax.set_yscale("linear")

        gui.rsm_ccnplot_ax.set_xticks([])
        gui.rsm_ccnplot_ax.set_yticks([])

        gui.rsm_ccnplot_cb.ax.set_yticks([])

        gui.rsm_ccnplot_ax.text(
            0.5,
            0.5,
            "Missing SOSAA RSM Predictions",
            color="red",
            size=20,
            ha="center",
            va="center",
            transform=gui.rsm_ccnplot_ax.transAxes,
        )
    else:
        level_mask = gui.rsm_prediction.index.get_level_values(1)
        level_heights = gui.rsm_prediction.index.levels[1]

        gui.rsm_ccnplot_ax.set_title(
            "Predicted CCN Concentration Profile"
            + "\nConfidence:"
            f" {np.mean(gui.rsm_prediction['log10_ccn_perturbed_conf']):.02}"
            + "\nWARNING: The SOSAA RSM is currently known to struggle"
            + " with small perturbations."
        )

        colours = mpl.cm.rainbow(np.linspace(0, 1, len(level_heights)))

        for lv, h in enumerate(level_heights):
            gui.rsm_ccnplot_ax.fill_between(
                gui.rsm_prediction[level_mask == h].index.get_level_values(0)
                / (60 * 60),
                np.power(
                    10.0,
                    gui.rsm_prediction[level_mask == h][
                        "log10_ccn_perturbed_pred"
                    ]
                    - gui.rsm_prediction[level_mask == h][
                        "log10_ccn_perturbed_stdv"
                    ],
                )
                - 1,
                np.power(
                    10.0,
                    gui.rsm_prediction[level_mask == h][
                        "log10_ccn_perturbed_pred"
                    ]
                    + gui.rsm_prediction[level_mask == h][
                        "log10_ccn_perturbed_stdv"
                    ],
                )
                - 1,
                color=colours[lv],
                alpha=0.35,
            )

        for lv, h in enumerate(level_heights):
            gui.rsm_ccnplot_ax.plot(
                gui.rsm_prediction[level_mask == h].index.get_level_values(0)
                / (60 * 60),
                np.power(
                    10.0,
                    gui.rsm_prediction[level_mask == h][
                        "log10_ccn_perturbed_pred"
                    ],
                )
                - 1,
                color=colours[lv],
            )

        gui.rsm_ccnplot_ax.set_yscale("log")

        gui.rsm_ccnplot_ax.set_xticks(
            [
                h
                for h in range(
                    int((gui.rsm_prediction.index.levels[0][0] // (60 * 60))),
                    0,
                    24,
                )
            ]
        )
        gui.rsm_ccnplot_ax.set_xticklabels(
            [
                (gui.rsm_dt + datetime.timedelta(hours=h)).strftime("%d.%m")
                for h in range(
                    int((gui.rsm_prediction.index.levels[0][0] // (60 * 60))),
                    0,
                    24,
                )
            ]
        )

        gui.rsm_ccnplot_cb.ax.set_yticks(np.linspace(0, 1, len(level_heights)))
        gui.rsm_ccnplot_cb.ax.set_yticklabels(
            [
                f"{int(h)}m" if i % 5 == 0 else ""
                for i, h in enumerate(level_heights)
            ]
        )

    gui.rsm_ccnplot_fig.tight_layout()
    gui.rsm_ccnplot_canvas.draw()
