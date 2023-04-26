import datetime

from PyQt5 import QtWidgets


def init_plot(gui):
    if getattr(gui, "rsm_diffplot_canvas", None) is not None:
        return

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )

    gui.rsm_diffplot_fig, gui.rsm_diffplot_ax = plt.subplots(
        1, 1, figsize=(6, 4), dpi=100
    )
    gui.rsm_diffplot_canvas = FigureCanvasQTAgg(gui.rsm_diffplot_fig)

    gui.rsm_diffplot_cb = gui.rsm_diffplot_fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=None,
            cmap="viridis",
        ),
        ax=gui.rsm_diffplot_ax,
        orientation="horizontal",
        extend="min",
    )

    diffplot_layout = QtWidgets.QVBoxLayout()
    diffplot_layout.addWidget(NavigationToolbar(gui.rsm_diffplot_canvas, gui))
    diffplot_layout.addWidget(gui.rsm_diffplot_canvas)

    gui.rsm_diffplot_tab.setLayout(diffplot_layout)


def update_plot(gui):
    import numpy as np

    gui.rsm_diffplot_ax.cla()

    gui.rsm_diffplot_ax.set_title(
        "Change in Cloud-Condensation-Nuclei (CCN) Concentration"
    )

    gui.rsm_diffplot_ax.set_xlabel("Baseline CCN concentration [m$^{-3}$]")
    gui.rsm_diffplot_ax.set_ylabel(
        "Perturbed Change in CCN concentration [m$^{-3}$]"
    )

    if gui.rsm_prediction is None:
        gui.rsm_diffplot_ax.set_xscale("linear")
        gui.rsm_diffplot_ax.set_yscale("linear")

        gui.rsm_diffplot_ax.set_xticks([])
        gui.rsm_diffplot_ax.set_yticks([])

        gui.rsm_diffplot_cb.ax.set_xticks([])

        gui.rsm_diffplot_ax.text(
            0.5,
            0.5,
            "Missing SOSAA RSM Predictions",
            color="red",
            size=20,
            ha="center",
            va="center",
            transform=gui.rsm_diffplot_ax.transAxes,
        )
    else:
        time = gui.rsm_prediction.index.get_level_values(0)
        log10_ccn_baseline = (
            gui.rsm_prediction["log10_ccn_baseline"].to_numpy().flatten()
        )
        log10_ccn_perturbed_pred = (
            gui.rsm_prediction["log10_ccn_perturbed_pred"].to_numpy().flatten()
        )
        log10_ccn_perturbed_stdv = (
            gui.rsm_prediction["log10_ccn_perturbed_stdv"].to_numpy().flatten()
        )
        log10_ccn_perturbed_conf = (
            gui.rsm_prediction["log10_ccn_perturbed_conf"].to_numpy().flatten()
        )

        gui.rsm_diffplot_ax.set_title(
            "Change in Cloud-Condensation-Nuclei (CCN) Concentration"
            + f"\nConfidence: {np.mean(log10_ccn_perturbed_conf):.02}"
            + "\nWARNING: The SOSAA RSM is currently known to struggle"
            + " with small perturbations."
        )

        for _ in range(gui.rsm_train_samples.value()):
            log10_ccn_perturbed = np.random.normal(
                loc=log10_ccn_perturbed_pred,
                scale=log10_ccn_perturbed_stdv,
            )

            I_conf = (
                np.random.random(size=log10_ccn_perturbed_conf.shape)
                <= log10_ccn_perturbed_conf
            )

            gui.rsm_diffplot_ax.scatter(
                np.power(10.0, log10_ccn_baseline[I_conf]),
                np.power(10.0, log10_ccn_perturbed[I_conf])
                - np.power(10.0, log10_ccn_baseline[I_conf]),
                c=time[I_conf],
                cmap="viridis",
                s=5,
            )

        xlim = gui.rsm_diffplot_ax.get_xlim()
        gui.rsm_diffplot_ax.plot(xlim, [0, 0], c="black", lw=1)
        gui.rsm_diffplot_ax.set_xlim(xlim)

        gui.rsm_diffplot_cb.ax.set_xticks(
            [
                (
                    (
                        h
                        - int(
                            (
                                gui.rsm_prediction.index.levels[0][0]
                                // (60 * 60)
                            )
                        )
                    )
                    / (
                        -int(
                            (
                                gui.rsm_prediction.index.levels[0][0]
                                // (60 * 60)
                            )
                        )
                    )
                )
                for h in range(
                    int((gui.rsm_prediction.index.levels[0][0] // (60 * 60))),
                    0,
                    24,
                )
            ]
        )
        gui.rsm_diffplot_cb.ax.set_xticklabels(
            [
                (gui.rsm_dt + datetime.timedelta(hours=h)).strftime("%d.%m")
                for h in range(
                    int((gui.rsm_prediction.index.levels[0][0] // (60 * 60))),
                    0,
                    24,
                )
            ]
        )

    gui.rsm_diffplot_fig.tight_layout()
    gui.rsm_diffplot_canvas.draw()
