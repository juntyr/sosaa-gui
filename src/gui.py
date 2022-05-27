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

import os

import darkdetect

from pathlib import Path

from PyQt5 import QtWidgets, QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication

from .config import get_config, set_config, remove_config
from .layouts import gui
from .resources import resource_path
from .settings import (
    print_settings,
    load_settings,
    save_settings,
    default_settings_path,
    minimal_settings_path,
)
from .style import hsl_to_hex, get_style_palette
from .version import sosaa_version_pretty
from .qt import operating_system


class QtSosaaGui(gui.Ui_MainWindow, QtWidgets.QMainWindow):
    """Main program window."""

    def __init__(self):
        super(QtSosaaGui, self).__init__()
        self.setupUi(self)

        self.setWindowTitle(sosaa_version_pretty)
        self.setWindowIcon(QtGui.QIcon(resource_path("icons/thebox_ico.png")))

        self.actionQuit_Ctrl_Q.triggered.connect(self.close)

        def buttonStyle(icon):
            icon_path_escaped = resource_path(icon).replace("\\", "\\\\")

            return f"background-image: url('{icon_path_escaped}');\nbackground-repeat: no-repeat;"

        self.saveCurrentButton.setStyleSheet(buttonStyle("icons/saveia.png"))
        self.saveButton.setStyleSheet(buttonStyle("icons/saveas.png"))
        self.loadButton.setStyleSheet(buttonStyle("icons/load.png"))
        self.saveDefaults.setStyleSheet(buttonStyle("icons/pack.png"))
        self.recompile.setStyleSheet(buttonStyle("icons/recompile.png"))

        self.currentInitFileToSave = None

        self.themeGroup = QtWidgets.QActionGroup(self)
        self.themeGroup.addAction(self.actionSystem)
        self.themeGroup.addAction(self.actionDark)
        self.themeGroup.addAction(self.actionLight)

        self.modelTypeGroup = QtWidgets.QButtonGroup(self)
        self.modelTypeGroup.addButton(self.flag_station_model)
        self.modelTypeGroup.addButton(self.flag_trajectory_model)

        def setStyleColour():
            colourSelector = QtWidgets.QColorDialog(self)

            colourSelector.setOption(
                QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog
            )
            colourSelector.setOption(QtWidgets.QColorDialog.ColorDialogOption.NoButtons)

            def colourChanged(colour):
                self.hue = colour.getHsl()[0]
                set_config("style", "hue", str(self.hue))
                setLightDarkStyle(self.dark)

            colourSelector.setCurrentColor(QtGui.QColor(hsl_to_hex(self.hue, 100, 100)))
            colourSelector.currentColorChanged.connect(colourChanged)

            colourSelector.exec_()

        self.actionChange_Colour.triggered.connect(setStyleColour)

        def setLightDarkStyle(dark):
            self.dark = dark

            QCoreApplication.instance().setPalette(
                get_style_palette(self.hue, self.dark)
            )

            # Force redraw the buttons
            self.saveCurrentButton.setStyleSheet(
                buttonStyle(
                    "icons/saveia.png"
                    if self.currentInitFileToSave is None
                    else "icons/save.png"
                )
            )
            self.saveButton.setStyleSheet(buttonStyle("icons/saveas.png"))
            self.loadButton.setStyleSheet(buttonStyle("icons/load.png"))
            self.saveDefaults.setStyleSheet(buttonStyle("icons/pack.png"))
            self.recompile.setStyleSheet(buttonStyle("icons/recompile.png"))

        def loadDefaultStyle():
            self.hue = int(get_config("style", "hue", fallback="316"))

            theme = get_config("style", "theme", fallback="system")

            if theme == "light":
                self.actionLight.setChecked(True)
                setLightDarkStyle(False)
            elif theme == "dark":
                self.actionDark.setChecked(True)
                setLightDarkStyle(True)
            else:
                self.actionSystem.setChecked(True)
                setLightDarkStyle(darkdetect.isDark())

        loadDefaultStyle()

        def actionSystemTrigger(checked):
            if checked:
                set_config("style", "theme", "system")
                setLightDarkStyle(darkdetect.isDark())

        self.actionSystem.triggered.connect(actionSystemTrigger)

        def actionLightTrigger(checked):
            if checked:
                set_config("style", "theme", "light")
                setLightDarkStyle(False)

        self.actionLight.triggered.connect(actionLightTrigger)

        def actionDarkTrigger(checked):
            if checked:
                set_config("style", "theme", "dark")
                setLightDarkStyle(True)

        self.actionDark.triggered.connect(actionDarkTrigger)

        def actionResetStyleTrigger():
            remove_config("style", "hue")
            remove_config("style", "theme")

            loadDefaultStyle()

        self.actionReset_Style.triggered.connect(actionResetStyleTrigger)

        def loadDefaultFont():
            font_str = get_config("style", "font", fallback=None)

            if font_str is None:
                font = QtGui.QFont()
                font.setBold(False)
                font.setItalic(False)
                font.setWeight(50)
            else:
                font = QtGui.QFont()
                font.fromString(font_str)

            self.setFont(font)

            self.menuFile.setFont(font)
            self.menuTools.setFont(font)
            self.menuSettings.setFont(font)
            self.menuStyle.setFont(font)
            self.menuFont.setFont(font)
            self.menuHelp.setFont(font)

        loadDefaultFont()

        def actionSetGlobalFontTrigger():
            fontSelector = QtWidgets.QFontDialog()
            font, ok = fontSelector.getFont(self.font(), parent=self)

            if ok:
                set_config("style", "font", font.toString())

                self.setFont(font)

                self.menuFile.setFont(font)
                self.menuTools.setFont(font)
                self.menuSettings.setFont(font)
                self.menuStyle.setFont(font)
                self.menuFont.setFont(font)
                self.menuHelp.setFont(font)

        self.actionSet_Global_Font.triggered.connect(actionSetGlobalFontTrigger)

        def actionResetFontTrigger():
            remove_config("style", "font")

            loadDefaultFont()

        self.actionReset_Fonts.triggered.connect(actionResetFontTrigger)

        def browsePath(title, save=False, directory=False, origin=None):
            pathSelector = QtWidgets.QFileDialog()

            pathSelector.setAcceptMode(
                pathSelector.AcceptMode.AcceptSave
                if save
                else pathSelector.AcceptMode.AcceptOpen
            )

            pathSelector.setOption(pathSelector.DontUseNativeDialog)

            if not save:
                pathSelector.setFileMode(pathSelector.FileMode.ExistingFile)

            if directory:
                pathSelector.setFileMode(pathSelector.FileMode.Directory)
                pathSelector.setOption(pathSelector.ShowDirsOnly)
                pathSelector.setOption(pathSelector.DontResolveSymlinks)

            pathSelector.setWindowTitle(title)

            pathSelector.setDirectory(origin or str(Path.cwd()))

            if pathSelector.exec() == 1:
                path = Path(pathSelector.selectedFiles()[0]).resolve()
            else:
                path = None

            return path

        def loadSettings():
            path = browsePath(title="Choose INITFILE")

            if (
                str(path) != default_settings_path
                and str(path) != minimal_settings_path
            ):
                self.currentInitFileToSave = path
            else:
                self.currentInitFileToSave = None

            if path is not None:
                load_settings(self, path)

            if self.currentInitFileToSave is None:
                self.hideCurrentInitFile()
            else:
                self.showCurrentInitFile(self.currentInitFileToSave)

        self.hideCurrentInitFile()

        self.actionOpen.triggered.connect(loadSettings)
        self.loadButton.clicked.connect(loadSettings)

        def loadMinimalSettings():
            self.currentInitFileToSave = None

            load_settings(self, minimal_settings_path)

            self.hideCurrentInitFile()

        self.actionLoad_minimal_settings.triggered.connect(loadMinimalSettings)

        if Path(default_settings_path).exists():
            load_settings(self, default_settings_path)
        else:
            load_settings(self, minimal_settings_path)

        self.actionPrint.triggered.connect(lambda: print_settings(self))

        def saveSettings(path=None):
            if path is None:
                path = browsePath(title="Save INITFILE", save=True)

            if path is None or str(path) == minimal_settings_path:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setText("Invalid INITFILE path")
                msg.setInformativeText(
                    "Overwriting the minimal INITFILE from the GUI is not allowed."
                )
                msg.setWindowTitle("Error saving INITFILE")
                msg.exec_()

                return

            save_settings(self, path)

            if str(path) != default_settings_path:
                self.currentInitFileToSave = path

                self.showCurrentInitFile(path)
            else:
                self.hideCurrentInitFile()

        self.saveButton.clicked.connect(lambda: saveSettings(path=None))
        self.actionSave_2.triggered.connect(lambda: saveSettings(path=None))

        self.saveCurrentButton.clicked.connect(
            lambda: saveSettings(path=self.currentInitFileToSave)
        )
        self.actionSave_to_current.triggered.connect(
            lambda: saveSettings(path=self.currentInitFileToSave)
        )

        self.saveDefaults.clicked.connect(
            lambda: saveSettings(path=default_settings_path)
        )

        self.setAcceptDrops(True)

        def timeZoneValueChanged(value):
            if value >= 0.0:
                self.time_zone.setPrefix("UTC+")
            else:
                self.time_zone.setPrefix("UTC")

        self.time_zone.valueChanged.connect(timeZoneValueChanged)

        def changeMainDirectory():
            path = browsePath(title="Choose the main directory", directory=True)

            if path is None:
                return

            if str(path).startswith(str(Path.cwd())):
                path = path.relative_to(Path.cwd())

            self.main_dir.setText(str(path))

        self.browse_main.clicked.connect(changeMainDirectory)

        def changeChemistryDirectory():
            main_dir = Path(self.main_dir.text()).resolve()

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

            self.chem_dir.setText(str(path))

        self.browse_chem.clicked.connect(changeChemistryDirectory)

        def changeCaseDirectory():
            main_dir = Path(self.main_dir.text()).resolve()

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

            self.case_dir.setText(str(path))

        self.browse_case.clicked.connect(changeCaseDirectory)

        def changeOutputDirectory():
            case_dir = Path(self.main_dir.text()).resolve() / self.case_dir.text()

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

            self.output_dir.setText(str(path))

        self.browse_output.clicked.connect(changeOutputDirectory)

        def changeInputDirectory():
            path = browsePath(title="Choose the input directory", directory=True)

            if path is None:
                return

            if str(path).startswith(str(Path.cwd())):
                path = path.relative_to(Path.cwd())

            self.input_dir.setText(str(path))

        self.browse_input.clicked.connect(changeInputDirectory)

        def changeStationDirectory():
            input_dir = Path(self.input_dir.text()).resolve()

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

            self.station.setText(str(path))
            self.station_name.setText(str(path.stem).capitalize())

        self.browse_station.clicked.connect(changeStationDirectory)

        def changeFlagEmis(flag_emis):
            self.dt_emis.setEnabled(flag_emis)
            self.label_dt_emis.setEnabled(flag_emis)
            self.group_output_list_emi.setEnabled(flag_emis)

        self.flag_emis.stateChanged.connect(changeFlagEmis)

        def changeFlagChem(flag_chem):
            self.dt_chem.setEnabled(flag_chem)
            self.label_dt_chem.setEnabled(flag_chem)
            self.label_chem_dir.setEnabled(flag_chem)
            self.chem_dir.setEnabled(flag_chem)
            self.browse_chem.setEnabled(flag_chem)

        self.flag_chem.clicked.connect(changeFlagChem)

        def changeFlagGasDryDep(flag_gasdrydep):
            self.dt_depo.setEnabled(flag_gasdrydep)
            self.label_dt_depo.setEnabled(flag_gasdrydep)
            self.group_output_list_Vd.setEnabled(flag_gasdrydep)

        self.flag_gasdrydep.stateChanged.connect(changeFlagGasDryDep)

        def changeFlagAero(flag_aero):
            self.dt_aero.setEnabled(flag_aero)
            self.label_dt_aero.setEnabled(flag_aero)
            self.dt_uhma.setEnabled(flag_aero)
            self.label_dt_uhma.setEnabled(flag_aero)
            self.aero_start_label.setEnabled(flag_aero)
            self.aero_start_offset.setEnabled(flag_aero)
            self.aero_start_date.setEnabled(flag_aero)

        self.flag_aero.clicked.connect(changeFlagAero)

        def changeModelType(_id, _checked):
            self.group_trajectory_model.setEnabled(
                self.flag_trajectory_model.isChecked()
            )

            if self.flag_trajectory_model.isChecked():
                self.time_zone.setValue(0.0)
                self.time_zone.setEnabled(False)

                fullDays = self.trajectory_duration.value()

                if fullDays < 0:
                    self.start_date.setEnabled(False)
                    self.end_date.setEnabled(True)

                    self.start_date.setDate(self.end_date.date().addDays(fullDays))
                    self.start_date.setTime(QtCore.QTime(0, 0))

                    self.aero_start_date.setText(
                        self.start_date.dateTime()
                        .addSecs(self.aero_start_offset.value() * 60 * 60)
                        .toString(" dd/MM/yyyy HH:mm:ss")
                    )
                else:
                    self.start_date.setEnabled(True)
                    self.end_date.setEnabled(False)

                    self.end_date.setDate(
                        self.start_date.date().addDays(
                            fullDays + (self.start_date.time() > QtCore.QTime(0, 0))
                        )
                    )
                    self.end_date.setTime(QtCore.QTime(0, 0))
            else:
                self.start_date.setEnabled(True)
                self.end_date.setEnabled(True)
                self.time_zone.setEnabled(True)

        self.modelTypeGroup.idToggled.connect(changeModelType)

        def changeTrajectoryDuration():
            fullDays = self.trajectory_duration.value()

            if fullDays < 0:
                self.start_date.setEnabled(False)
                self.end_date.setEnabled(True)

                self.start_date.setDate(self.end_date.date().addDays(fullDays))
                self.start_date.setTime(QtCore.QTime(0, 0))

                self.aero_start_date.setText(
                    self.start_date.dateTime()
                    .addSecs(self.aero_start_offset.value() * 60 * 60)
                    .toString(" dd/MM/yyyy HH:mm:ss")
                )
            else:
                self.start_date.setEnabled(True)
                self.end_date.setEnabled(False)

                self.end_date.setDate(
                    self.start_date.date().addDays(
                        fullDays + (self.start_date.time() > QtCore.QTime(0, 0))
                    )
                )
                self.end_date.setTime(QtCore.QTime(0, 0))

        self.trajectory_duration.valueChanged.connect(changeTrajectoryDuration)

        def changeAerosolStartOffset():
            self.aero_start_date.setText(
                self.start_date.dateTime()
                .addSecs(self.aero_start_offset.value() * 60 * 60)
                .toString(" dd/MM/yyyy HH:mm:ss")
            )

        self.aero_start_offset.valueChanged.connect(changeAerosolStartOffset)

        def changeStartDate():
            if self.flag_trajectory_model.isChecked():
                fullDays = self.trajectory_duration.value()

                if fullDays >= 0:
                    self.end_date.setDate(
                        self.start_date.date().addDays(
                            fullDays + (self.start_date.time() > QtCore.QTime(0, 0))
                        )
                    )
                    self.end_date.setTime(QtCore.QTime(0, 0))

            self.aero_start_date.setText(
                self.start_date.dateTime()
                .addSecs(self.aero_start_offset.value() * 60 * 60)
                .toString(" dd/MM/yyyy HH:mm:ss")
            )

        self.start_date.dateTimeChanged.connect(changeStartDate)

        def changeEndDate():
            if self.flag_trajectory_model.isChecked():
                fullDays = self.trajectory_duration.value()

                if fullDays < 0:
                    self.start_date.setDate(self.end_date.date().addDays(fullDays))
                    self.start_date.setTime(QtCore.QTime(0, 0))

                    self.aero_start_date.setText(
                        self.start_date.dateTime()
                        .addSecs(self.aero_start_offset.value() * 60 * 60)
                        .toString(" dd/MM/yyyy HH:mm:ss")
                    )

        self.end_date.dateTimeChanged.connect(changeEndDate)

        def openOutputDirectory():
            output_dir = (
                Path(self.main_dir.text()).resolve()
                / self.case_dir.text()
                / self.output_dir.text()
            )

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

        self.actionOpen_output_directory.triggered.connect(openOutputDirectory)

        def createOutputDirectories():
            output_dir = (
                Path(self.main_dir.text()).resolve()
                / self.case_dir.text()
                / self.output_dir.text()
            )

            output_dir_str = str(output_dir)

            if output_dir_str.startswith(str(Path.cwd())):
                output_dir_str = f"./{output_dir.relative_to(Path.cwd())}"

            try:
                output_dir.mkdir(parents=True)

                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText(f"The output directory '{output_dir_str}' was created.")
                msg.setWindowTitle("Created OUTPUT directory")
                msg.exec_()
            except FileExistsError:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setText(f"The output directory '{output_dir_str}' already existed.")
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

        self.actionCreate_output_directories.triggered.connect(createOutputDirectories)

    def showCurrentInitFile(self, path):
        def buttonStyle(icon):
            icon_path_escaped = resource_path(icon).replace("\\", "\\\\")

            return f"background-image: url('{icon_path_escaped}');\nbackground-repeat: no-repeat;"

        self.saveCurrentButton.setStyleSheet(buttonStyle("icons/save.png"))
        self.saveCurrentButton.setEnabled(True)
        self.actionSave_to_current.setEnabled(True)

        if str(path).startswith(str(Path.cwd())):
            path = path.relative_to(Path.cwd())

        self.currentInitFile.setText(str(path))

        self.setWindowTitle(f"{sosaa_version_pretty}: {path}")
        self.saveCurrentButton.setToolTip(f"Save to {path.name}")

    def hideCurrentInitFile(self):
        def buttonStyle(icon):
            icon_path_escaped = resource_path(icon).replace("\\", "\\\\")

            return f"background-image: url('{icon_path_escaped}');\nbackground-repeat: no-repeat;"

        self.saveCurrentButton.setEnabled(False)
        self.actionSave_to_current.setEnabled(False)
        self.saveCurrentButton.setStyleSheet(buttonStyle("icons/saveia.png"))

        self.currentInitFile.setText("")

        self.setWindowTitle(sosaa_version_pretty)
        self.saveCurrentButton.setToolTip("")

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat("text/plain"):
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        path = Path(e.mimeData().text().replace("file://", "").strip()).resolve()

        if str(path) != default_settings_path and str(path) != minimal_settings_path:
            self.currentInitFileToSave = path
        else:
            self.currentInitFileToSave = None

        load_settings(self, path)

        if self.currentInitFileToSave is None:
            self.hideCurrentInitFile()
        else:
            self.showCurrentInitFile(self.currentInitFileToSave)
