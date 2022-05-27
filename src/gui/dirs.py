#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
Copyright (C) 2022 Multi-Scale Modelling group
Institute for Atmospheric and Earth System Research (INAR),
University of Helsinki
Contact information sosaa@helsinki.fi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
=============================================================================
"""

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

    def changeStationDirectory():
        input_dir = Path(gui.input_dir.text()).resolve()

        path = browsePath(
            title="Choose the station directory",
            directory=True,
            origin=str(input_dir),
        )

        if path is None:
            return

        if not str(path).startswith(str(input_dir)):
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Invalid station directory")
            msg.setInformativeText(
                "The station directory must be inside the input directory."
            )
            msg.setWindowTitle("Error selecting directory")
            msg.exec_()

            return

        path = path.relative_to(input_dir)

        gui.station.setText(str(path))
        gui.station_name.setText(str(path.stem).capitalize())

    gui.browse_station.clicked.connect(changeStationDirectory)
