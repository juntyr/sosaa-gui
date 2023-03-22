import os
from pathlib import Path

from PyQt5 import QtCore, QtWidgets


def init_run_gui(gui):
    gui.model_start.setEnabled(True)
    gui.model_stop.setEnabled(False)

    def startModelRun():
        terminal = gui.terminal_run

        gui.model_start.setEnabled(False)
        gui.model_stop.setEnabled(True)

        if getattr(terminal, "process", None) is not None:
            return

        if not gui.saveCurrentButton.isEnabled():
            gui.model_start.setEnabled(True)
            gui.model_stop.setEnabled(False)

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Unsaved INITFILE")
            msg.setInformativeText(
                "The INITFILE must be saved before starting a new SOSAA run."
            )
            msg.setWindowTitle("Error starting SOSAA")
            msg.exec_()

            return

        gui.actionSave_to_current.trigger()

        terminal.process = QtCore.QProcess(terminal)
        terminal.process.start(
            "urxvt",
            [
                "-embed",
                str(int(terminal.winId())),
                "+sb",
                "-hold",
                "-e",
                os.environ.get("SHELL", "sh"),
                "-x",
                "-c",
                " ".join(
                    [
                        "cd",
                        str(
                            Path(gui.main_dir.text()).resolve()
                            / gui.case_dir.text()
                            / gui.casename_dir.text()
                        ),
                        "&&",
                        f"./{gui.compile_exe.text()}",
                        str(Path(gui.currentInitFile.text()).resolve()),
                    ]
                ),
            ],
        )

    gui.model_start.clicked.connect(startModelRun)

    def forceStopModelRun():
        terminal = gui.terminal_run

        gui.model_start.setEnabled(True)
        gui.model_stop.setEnabled(False)

        if getattr(terminal, "process", None) is None:
            return

        terminal.process.kill()
        terminal.process = None

    gui.model_stop.clicked.connect(forceStopModelRun)
