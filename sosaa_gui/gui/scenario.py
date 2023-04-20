from PyQt5 import QtCore

from ..settings import is_loading


def init_scenario_gui(gui):
    def changeTrajectoryDuration():
        if is_loading():
            return

        lastFullDays = gui.trajectory_duration.lastValue
        fullDays = gui.trajectory_duration.value()

        gui.trajectory_duration.lastValue = fullDays

        if fullDays < 0:
            # Negative duration: start_date = floor(end_date - full_days)
            gui.start_date.setEnabled(False)
            gui.end_date.setEnabled(True)

            if lastFullDays > 0:
                gui.end_date.setDate(gui.start_date.date())

            gui.start_date.setDate(gui.end_date.date().addDays(fullDays))
            gui.start_date.setTime(QtCore.QTime(0, 0))

            gui.aero_start_date.setText(
                gui.start_date.dateTime()
                .addSecs(gui.aero_start_offset.value() * 60 * 60)
                .toString(" dd/MM/yyyy HH:mm:ss")
            )
        elif fullDays > 0:
            # Positive duration: end_date = ceil(start_date + full_days)
            gui.start_date.setEnabled(True)
            gui.end_date.setEnabled(False)

            if lastFullDays < 0:
                gui.start_date.setDate(gui.end_date.date())

            gui.end_date.setDate(
                gui.start_date.date().addDays(
                    fullDays + (gui.start_date.time() > QtCore.QTime(0, 0))
                )
            )
            gui.end_date.setTime(QtCore.QTime(0, 0))
        else:
            # Zero duration, check which direction we are coming from

            if lastFullDays > 0:
                gui.end_date.setDate(
                    gui.start_date.date().addDays(
                        gui.start_date.time() > QtCore.QTime(0, 0)
                    )
                )
            else:
                gui.start_date.setDate(gui.end_date.date())

    gui.trajectory_duration.valueChanged.connect(changeTrajectoryDuration)
    gui.trajectory_duration.lastValue = gui.trajectory_duration.value()

    def changeAerosolStartOffset():
        if is_loading():
            return

        gui.aero_start_date.setText(
            gui.start_date.dateTime()
            .addSecs(gui.aero_start_offset.value() * 60 * 60)
            .toString(" dd/MM/yyyy HH:mm:ss")
        )

    gui.aero_start_offset.valueChanged.connect(changeAerosolStartOffset)

    def changeStartDate():
        if is_loading():
            return

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
        if is_loading():
            return

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
