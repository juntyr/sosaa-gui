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

from PyQt5 import QtWidgets, QtCore


def init_scenario_gui(gui):
    # Model type can be EITHER station or trajectory
    gui.modelTypeGroup = QtWidgets.QButtonGroup(gui)
    gui.modelTypeGroup.addButton(gui.flag_station_model)
    gui.modelTypeGroup.addButton(gui.flag_trajectory_model)

    def timeZoneValueChanged(value):
        if value >= 0.0:
            gui.time_zone.setPrefix("UTC+")
        else:
            gui.time_zone.setPrefix("UTC")

    gui.time_zone.valueChanged.connect(timeZoneValueChanged)

    def changeModelType(_id, _checked):
        # Fake-access the _id and _checked variables
        _id = _id
        _checked = _checked

        gui.group_trajectory_model.setEnabled(gui.flag_trajectory_model.isChecked())

        if gui.flag_trajectory_model.isChecked():
            # In trajectory mode all times are in UTC+0
            gui.time_zone.setValue(0.0)
            gui.time_zone.setEnabled(False)

            fullDays = gui.trajectory_duration.value()

            if fullDays < 0:
                # Negative duration: start_date = floor(end_date - full_days)
                gui.start_date.setEnabled(False)
                gui.end_date.setEnabled(True)

                gui.start_date.setDate(gui.end_date.date().addDays(fullDays))
                gui.start_date.setTime(QtCore.QTime(0, 0))

                gui.aero_start_date.setText(
                    gui.start_date.dateTime()
                    .addSecs(gui.aero_start_offset.value() * 60 * 60)
                    .toString(" dd/MM/yyyy HH:mm:ss")
                )
            else:
                # Positive duration: end_date = ceil(start_date + full_days)
                gui.start_date.setEnabled(True)
                gui.end_date.setEnabled(False)

                gui.end_date.setDate(
                    gui.start_date.date().addDays(
                        fullDays + (gui.start_date.time() > QtCore.QTime(0, 0))
                    )
                )
                gui.end_date.setTime(QtCore.QTime(0, 0))
        else:
            gui.start_date.setEnabled(True)
            gui.end_date.setEnabled(True)
            gui.time_zone.setEnabled(True)

    gui.modelTypeGroup.idToggled.connect(changeModelType)

    def changeTrajectoryDuration():
        fullDays = gui.trajectory_duration.value()

        if fullDays < 0:
            # Negative duration: start_date = floor(end_date - full_days)
            gui.start_date.setEnabled(False)
            gui.end_date.setEnabled(True)

            gui.start_date.setDate(gui.end_date.date().addDays(fullDays))
            gui.start_date.setTime(QtCore.QTime(0, 0))

            gui.aero_start_date.setText(
                gui.start_date.dateTime()
                .addSecs(gui.aero_start_offset.value() * 60 * 60)
                .toString(" dd/MM/yyyy HH:mm:ss")
            )
        else:
            # Positive duration: end_date = ceil(start_date + full_days)
            gui.start_date.setEnabled(True)
            gui.end_date.setEnabled(False)

            gui.end_date.setDate(
                gui.start_date.date().addDays(
                    fullDays + (gui.start_date.time() > QtCore.QTime(0, 0))
                )
            )
            gui.end_date.setTime(QtCore.QTime(0, 0))

    gui.trajectory_duration.valueChanged.connect(changeTrajectoryDuration)

    def changeAerosolStartOffset():
        gui.aero_start_date.setText(
            gui.start_date.dateTime()
            .addSecs(gui.aero_start_offset.value() * 60 * 60)
            .toString(" dd/MM/yyyy HH:mm:ss")
        )

    gui.aero_start_offset.valueChanged.connect(changeAerosolStartOffset)

    def changeStartDate():
        if gui.flag_trajectory_model.isChecked():
            fullDays = gui.trajectory_duration.value()

            if fullDays >= 0:
                # Positive duration: end_date = ceil(start_date + full_days)
                gui.end_date.setDate(
                    gui.start_date.date().addDays(
                        fullDays + (gui.start_date.time() > QtCore.QTime(0, 0))
                    )
                )
                gui.end_date.setTime(QtCore.QTime(0, 0))

        gui.aero_start_date.setText(
            gui.start_date.dateTime()
            .addSecs(gui.aero_start_offset.value() * 60 * 60)
            .toString(" dd/MM/yyyy HH:mm:ss")
        )

    gui.start_date.dateTimeChanged.connect(changeStartDate)

    def changeEndDate():
        if gui.flag_trajectory_model.isChecked():
            fullDays = gui.trajectory_duration.value()

            if fullDays < 0:
                # Negative duration: start_date = floor(end_date - full_days)
                gui.start_date.setDate(gui.end_date.date().addDays(fullDays))
                gui.start_date.setTime(QtCore.QTime(0, 0))

                gui.aero_start_date.setText(
                    gui.start_date.dateTime()
                    .addSecs(gui.aero_start_offset.value() * 60 * 60)
                    .toString(" dd/MM/yyyy HH:mm:ss")
                )

    gui.end_date.dateTimeChanged.connect(changeEndDate)
