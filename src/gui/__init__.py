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

from PyQt5 import QtWidgets, QtGui

from ..layouts import gui
from ..resources import resource_path
from ..version import sosaa_version_pretty
from .style import init_gui_style
from .loadsave import init_gui_loadsave
from .dirs import init_dirs_gui
from .modules import init_modules_gui
from .output import init_gui_output
from .scenario import init_scenario_gui


class QtSosaaGui(gui.Ui_MainWindow, QtWidgets.QMainWindow):
    """Main program window."""

    def __init__(self):
        super(QtSosaaGui, self).__init__()
        self.setupUi(self)

        self.currentInitFileToSave = None

        self.setWindowTitle(sosaa_version_pretty)
        self.setWindowIcon(QtGui.QIcon(resource_path("icons/thebox_ico.png")))

        self.actionQuit_Ctrl_Q.triggered.connect(self.close)

        init_gui_style(self)
        init_gui_loadsave(self)
        init_dirs_gui(self)
        init_modules_gui(self)
        init_scenario_gui(self)
        init_gui_output(self)
