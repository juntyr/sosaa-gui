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

import signal
import time

# Enable GUI termination using ctrl-c
signal.signal(signal.SIGINT, signal.SIG_DFL)

from PyQt5 import QtWidgets

from src.gui import QtSosaaGui
from src.qt import setup_qt_scaling, setup_qt_style
from src.version import sosaa_version, sosaa_version_pretty


__version__ = sosaa_version


if __name__ == "__main__":
    print(
        f"{sosaa_version_pretty} started at: {time.strftime('%B %d %Y, %H:%M:%S', time.localtime())}",
        flush=True,
    )

    setup_qt_scaling()

    app = QtWidgets.QApplication([])

    gui = QtSosaaGui()
    gui.setGeometry(30, 30, 900, 700)
    gui.show()

    setup_qt_style()

    app.exec_()
