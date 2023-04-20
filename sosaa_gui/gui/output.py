import os
from pathlib import Path

from PyQt5 import QtWidgets

from ..qt import operating_system


def init_gui_output(gui):
    def openOutputDirectory():
        output_dir = Path(gui.output_dir.text()).resolve()

        if not output_dir.exists():
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)

            if str(output_dir).startswith(str(Path.cwd())):
                output_dir = f"./{output_dir.relative_to(Path.cwd())}"

            msg.setText(f"The output directory '{output_dir}' does not exist.")
            msg.setWindowTitle("Invalid OUTPUT directory")
            msg.exec_()

            return

        if operating_system == "Windows":
            os.startfile(output_dir)
        if operating_system == "Linux":
            os.system(f'xdg-open "{output_dir}"')
        if operating_system == "Darwin":
            os.system('open "{output_dir}"')
        else:
            return

    gui.actionOpen_output_directory.triggered.connect(openOutputDirectory)

    def createOutputDirectories():
        output_dir = Path(gui.output_dir.text()).resolve()

        output_dir_str = str(output_dir)

        if output_dir_str.startswith(str(Path.cwd())):
            output_dir_str = f"./{output_dir.relative_to(Path.cwd())}"

        try:
            output_dir.mkdir(parents=True)

            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText(
                f"The output directory '{output_dir_str}' was created."
            )
            msg.setWindowTitle("Created OUTPUT directory")
            msg.exec_()
        except FileExistsError:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText(
                f"The output directory '{output_dir_str}' already existed."
            )
            msg.setWindowTitle("Created OUTPUT directory")
            msg.exec_()
        except Exception as e:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText(
                f"Failed to create the output directory '{output_dir_str}'."
            )
            msg.setInformativeText(str(e))
            msg.setWindowTitle("Invalid OUTPUT directory")
            msg.exec_()

    gui.actionCreate_output_directories.triggered.connect(
        createOutputDirectories
    )
