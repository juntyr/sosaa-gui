from pathlib import Path

from PyQt5 import QtWidgets

from .browse import browsePath


def init_dirs_gui(gui):
    def changeMainDirectory():
        path = browsePath(title="Choose the main directory", directory=True)

        if path is None:
            return

        if str(path).startswith(str(Path.cwd())):
            path = f"./{path.relative_to(Path.cwd())}"

        gui.main_dir.setText(str(path))

    gui.browse_main.clicked.connect(changeMainDirectory)

    def changeChemistryDirectory():
        main_dir = Path(gui.main_dir.text()).resolve()

        path = browsePath(
            title="Choose the chemistry directory",
            directory=True,
            origin=str(main_dir),
        )

        if path is None:
            return

        if not str(path).startswith(str(main_dir)):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Invalid chemistry directory")
            msg.setInformativeText(
                "The chemistry directory must be inside the main directory."
            )
            msg.setWindowTitle("Error selecting directory")
            msg.exec_()

            return

        path = path.relative_to(main_dir)

        gui.chem_dir.setText(str(path))

    gui.browse_chem.clicked.connect(changeChemistryDirectory)

    def changeCaseDirectory():
        main_dir = Path(gui.main_dir.text()).resolve()

        path = browsePath(
            title="Choose the case directory", directory=True, origin=str(main_dir)
        )

        if path is None:
            return

        if not str(path).startswith(str(main_dir)):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Invalid case directory")
            msg.setInformativeText(
                "The case directory must be inside the main directory."
            )
            msg.setWindowTitle("Error selecting directory")
            msg.exec_()

            return

        path = path.relative_to(main_dir)

        gui.case_dir.setText(str(path))

    gui.browse_case.clicked.connect(changeCaseDirectory)

    def changeOutputDirectory():
        case_dir = Path(gui.main_dir.text()).resolve() / gui.case_dir.text()

        path = browsePath(
            title="Choose the output directory",
            directory=True,
            origin=str(case_dir),
        )

        if path is None:
            return

        if not str(path).startswith(str(case_dir)):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Invalid output directory")
            msg.setInformativeText(
                "The output directory must be inside the case directory."
            )
            msg.setWindowTitle("Error selecting directory")
            msg.exec_()

            return

        path = path.relative_to(case_dir)

        gui.output_dir.setText(str(path))

    gui.browse_output.clicked.connect(changeOutputDirectory)

    def changeInputDirectory():
        path = browsePath(title="Choose the input directory", directory=True)

        if path is None:
            return

        if str(path).startswith(str(Path.cwd())):
            path = f"./{path.relative_to(Path.cwd())}"

        gui.input_dir.setText(str(path))

    gui.browse_input.clicked.connect(changeInputDirectory)
