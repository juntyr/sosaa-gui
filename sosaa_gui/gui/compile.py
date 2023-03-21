import os
from pathlib import Path

from PyQt5 import QtCore, QtWidgets


def init_compile_gui(gui):
    gui.compile_start.setEnabled(True)
    gui.compile_clean.setEnabled(True)
    gui.compile_cleanchem.setEnabled(True)
    gui.compile_stop.setEnabled(False)

    def startCompilation():
        terminal = gui.terminal_compile

        gui.compile_start.setEnabled(False)
        gui.compile_clean.setEnabled(False)
        gui.compile_cleanchem.setEnabled(False)
        gui.compile_stop.setEnabled(True)

        if getattr(terminal, "process", None) is not None:
            return

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
                        str(Path(gui.main_dir.text()).resolve() / gui.code_dir.text()),
                        "&&",
                        "make",
                        f"SOSAA_ROOT={Path(gui.main_dir.text()).resolve()}",
                        f"CODE_DIR={Path(gui.main_dir.text()).resolve() / gui.code_dir.text()}",
                        # f"CHEMALL_DIR={???}",
                        f"CASE_DIR={Path(gui.main_dir.text()).resolve() / gui.case_dir.text()}",
                        # f"CHEM={???}",
                        # f"CASE={???}",
                    ]
                    + (
                        [f"ALT_NAME={gui.compile_exe.text()}"]
                        if len(gui.compile_exe.text()) > 0
                        else []
                    )
                    + (
                        [f"INIT_FILE={Path(gui.currentInitFile.text()).resolve()}"]
                        if len(gui.currentInitFile.text()) > 0
                        else []
                    )
                ),
            ],
        )

    gui.compile_start.clicked.connect(startCompilation)

    def stopCompilation():
        terminal = gui.terminal_compile

        gui.compile_start.setEnabled(True)
        gui.compile_clean.setEnabled(True)
        gui.compile_cleanchem.setEnabled(True)
        gui.compile_stop.setEnabled(False)

        if getattr(terminal, "process", None) is None:
            return

        terminal.process.kill()
        terminal.process = None

    gui.compile_stop.clicked.connect(stopCompilation)

    def cleanSosaa():
        terminal = gui.terminal_compile

        gui.compile_start.setEnabled(False)
        gui.compile_clean.setEnabled(False)
        gui.compile_cleanchem.setEnabled(False)
        gui.compile_stop.setEnabled(True)

        if getattr(terminal, "process", None) is not None:
            return

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
                        str(Path(gui.main_dir.text()).resolve() / gui.code_dir.text()),
                        "&&",
                        "make",
                        "clean",
                    ]
                ),
            ],
        )

    gui.compile_clean.clicked.connect(cleanSosaa)

    def cleanChemistry():
        terminal = gui.terminal_compile

        gui.compile_start.setEnabled(False)
        gui.compile_clean.setEnabled(False)
        gui.compile_cleanchem.setEnabled(False)
        gui.compile_stop.setEnabled(True)

        if getattr(terminal, "process", None) is not None:
            return

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
                        str(Path(gui.main_dir.text()).resolve() / gui.code_dir.text()),
                        "&&",
                        "make",
                        "cleanchem",
                    ]
                ),
            ],
        )

    gui.compile_cleanchem.clicked.connect(cleanChemistry)

    def recompile():
        gui.tabWidget.setCurrentWidget(
            gui.tabWidget.findChild(QtWidgets.QWidget, "compile_tab")
        )

        startCompilation()

    gui.recompile.clicked.connect(recompile)
