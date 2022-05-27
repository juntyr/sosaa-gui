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
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication

from .resources import open_resource

try:
    import platform

    # "Windows" / "Linux" / "Darwin"
    operating_system = platform.system() or "Linux"
except:
    operating_system = "Linux"


# Set the GUI scaling factor for QT based on the operating system
def setup_qt_scaling():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    # Enable QT highdpi scaling and highdpi icons
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # Check if scaling is necessary, currently only required on Windows
    args = []

    for arg in sys.argv:
        args.append(arg.upper())

        if "--scaling_" in arg:
            os.environ[
                "QT_SCALE_FACTOR"
            ] = f"{float(arg.replace('--scaling_', '')):3.2f}"

            args.append("-NS")

    # FIXME petri: check if platform usually works for everyone
    if (os.name.upper() == "NT" or operating_system == "Windows") and not "-NS" in args:
        try:
            import ctypes

            sf = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
            if sf < 1.3:
                sf = 1
            sf = 1 / sf

            u32 = ctypes.windll.user32
            scrhgt = u32.GetSystemMetrics(1)

            if scrhgt < 850:
                sf = sf * scrhgt / 850.0
        except:
            sf = 1

            print(
                f"Failed to get the scaling factor of the screen, falling back to {sf:3.2f}."
            )

        os.environ["QT_SCALE_FACTOR"] = f"{sf:3.2f}"

    if "-NS" in args:
        print(
            f"Scaling factor overriden to {os.environ['QT_SCALE_FACTOR']} from the commandline."
        )


# Set the QT style for the GUI application
def setup_qt_style():
    styles = QtWidgets.QStyleFactory.keys()

    if "Fusion" in styles:
        QCoreApplication.instance().setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    else:
        print(f"Available styles: {styles}")
