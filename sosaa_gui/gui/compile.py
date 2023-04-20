from pathlib import Path

from PyQt5 import QtCore

from .terminal import generateTerminalCommand


def init_compile_gui(gui):
    gui.compile_start.setEnabled(True)
    gui.compile_clean.setEnabled(True)
    gui.compile_cleanchem.setEnabled(True)
    gui.compile_stop.setEnabled(False)

    def generateMakeVariables():
        return (
            [
                # fmt: off
                f"SOSAA_ROOT={Path(gui.main_dir.text()).resolve()}",
                f"CODE_DIR={Path(gui.main_dir.text()).resolve() / gui.code_dir.text()}",  # noqa: E501
                f"CHEMALL_DIR={Path(gui.main_dir.text()).resolve() / gui.chem_dir.text()}",  # noqa: E501
                f"CASE_DIR={Path(gui.main_dir.text()).resolve() / gui.case_dir.text()}",  # noqa: E501
                f"CHEM={gui.chemname_dir.text()}",
                f"CASE={gui.casename_dir.text()}",
                # fmt: on
            ]
            + (
                [f"ALT_NAME={gui.sosaa_exe.text()}"]
                if len(gui.sosaa_exe.text()) > 0
                else []
            )
            + (
                [f"INIT_FILE={Path(gui.currentInitFile.text()).resolve()}"]
                if len(gui.currentInitFile.text()) > 0
                else []
            )
        )

    def startCompilation():
        terminal = gui.terminal_compile

        gui.compile_start.setEnabled(False)
        gui.compile_clean.setEnabled(False)
        gui.compile_cleanchem.setEnabled(False)
        gui.compile_stop.setEnabled(True)

        if getattr(terminal, "process", None) is not None:
            return

        command = generateTerminalCommand(
            " ".join(
                [
                    "cd",
                    str(
                        Path(gui.main_dir.text()).resolve()
                        / gui.code_dir.text()
                    ),
                    "&&",
                    "make",
                    "sosaa.exe",
                ]
                + generateMakeVariables()
            ),
            int(terminal.winId()),
        )

        if command is None:
            gui.compile_start.setEnabled(True)
            gui.compile_clean.setEnabled(True)
            gui.compile_cleanchem.setEnabled(True)
            gui.compile_stop.setEnabled(False)

            return

        terminal.process = QtCore.QProcess(terminal)
        terminal.process.start(*command)

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

        command = generateTerminalCommand(
            " ".join(
                [
                    "cd",
                    str(
                        Path(gui.main_dir.text()).resolve()
                        / gui.code_dir.text()
                    ),
                    "&&",
                    "make",
                    "clean",
                ]
                + generateMakeVariables()
            ),
            int(terminal.winId()),
        )

        if command is None:
            gui.compile_start.setEnabled(True)
            gui.compile_clean.setEnabled(True)
            gui.compile_cleanchem.setEnabled(True)
            gui.compile_stop.setEnabled(False)

            return

        terminal.process = QtCore.QProcess(terminal)
        terminal.process.start(*command)

    gui.compile_clean.clicked.connect(cleanSosaa)

    def cleanChemistry():
        terminal = gui.terminal_compile

        gui.compile_start.setEnabled(False)
        gui.compile_clean.setEnabled(False)
        gui.compile_cleanchem.setEnabled(False)
        gui.compile_stop.setEnabled(True)

        if getattr(terminal, "process", None) is not None:
            return

        command = generateTerminalCommand(
            " ".join(
                [
                    "cd",
                    str(
                        Path(gui.main_dir.text()).resolve()
                        / gui.code_dir.text()
                    ),
                    "&&",
                    "make",
                    "cleanchem",
                ]
                + generateMakeVariables()
            ),
            int(terminal.winId()),
        )

        if command is None:
            gui.compile_start.setEnabled(True)
            gui.compile_clean.setEnabled(True)
            gui.compile_cleanchem.setEnabled(True)
            gui.compile_stop.setEnabled(False)

            return

        terminal.process = QtCore.QProcess(terminal)
        terminal.process.start(*command)

    gui.compile_cleanchem.clicked.connect(cleanChemistry)

    def recompile():
        gui.tabWidget.setCurrentWidget(gui.sosaa_tab)
        gui.sosaa_subtab.setCurrentWidget(gui.compile_tab)

        startCompilation()

    gui.recompile.clicked.connect(recompile)
    gui.actionRecompile_model.triggered.connect(recompile)
