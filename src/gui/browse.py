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
