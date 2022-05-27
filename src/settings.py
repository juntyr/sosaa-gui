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
import re
import time

import f90nml

from f90nml.namelist import Namelist

from PyQt5 import QtCore
from PyQt5 import QtWidgets

from .config import get_config, set_config
from .resources import resource_path


_raw_settings_header = f"! \\/ {'Raw input from the SOSAA GUI'.center(70, '-')} \\/ !"
_raw_settings_footer = f"! /\\ {'Raw input from the SOSAA GUI'.center(70, '-')} /\\ !"
_raw_setting_pattern = re.compile(
    rf"^{re.escape(_raw_settings_header)}$\n(.*?)\n^{re.escape(_raw_settings_footer)}$",
    flags=(re.MULTILINE | re.DOTALL),
)


_settings = Namelist()
_settings.indent = "  "
_settings.end_comma = True
_settings.uppercase = True

# TODO juniper: only change the minor version when something changes
_version_major = int(get_config("settings", "version", fallback="0")) + 1
set_config("settings", "version", str(_version_major))
_version_minor = 0


default_settings_path = resource_path("conf/defaults.init")
minimal_settings_path = resource_path("conf/minimal.init")


def _update_gui_from_main_settings(gui):
    main = _settings.get("nml_main", dict())

    main_dir = main.get("work_dir")
    if main_dir.startswith(str(Path.cwd())):
        main_dir = str(Path(main_dir).relative_to(Path.cwd()))
    gui.main_dir.setText(main_dir)

    gui.chem_dir.setText(main.get("chem_dir"))
    gui.case_dir.setText(main.get("case_dir"))

    input_dir = main.get("input_dir")
    if input_dir.startswith(str(Path.cwd())):
        input_dir = str(Path(input_dir).relative_to(Path.cwd()))
    gui.input_dir.setText(input_dir)

    gui.station.setText(main.get("station"))
    gui.station_name.setText(Path(main.get("station")).stem.capitalize())
    gui.output_dir.setText(main.get("output_dir"))


def _update_gui_from_flag_settings(gui):
    flag = _settings.get("nml_flag", dict())
    aer = _settings.get("aer_flag", dict()).get("options", dict())

    flag_emis = flag.get("flag_emis", 2) != 0
    gui.flag_emis.setChecked(flag_emis)
    gui.dt_emis.setEnabled(flag_emis)
    gui.label_dt_emis.setEnabled(flag_emis)
    gui.group_output_list_emi.setEnabled(flag_emis)

    # Note: Soil emissions are currently always disabled
    gui.flag_emis_soil.setChecked(False)

    flag_chem = flag.get("flag_chem", 1) != 0
    gui.flag_chem.setChecked(flag_chem)
    gui.dt_chem.setEnabled(flag_chem)
    gui.label_dt_chem.setEnabled(flag_chem)
    gui.label_chem_dir.setEnabled(flag_chem)
    gui.chem_dir.setEnabled(flag_chem)
    gui.browse_chem.setEnabled(flag_chem)

    flag_mix_chem = flag.get("flag_mix_chem", 1) != 0
    gui.flag_mix_chem.setChecked(flag_mix_chem)

    flag_gasdrydep = flag.get("flag_gasdrydep", 0) != 0
    gui.flag_gasdrydep.setChecked(flag_gasdrydep)
    gui.dt_depo.setEnabled(flag_gasdrydep)
    gui.label_dt_depo.setEnabled(flag_gasdrydep)
    gui.group_output_list_Vd.setEnabled(flag_gasdrydep)

    flag_aero = flag.get("flag_aero", 1) != 0
    gui.flag_aero.setChecked(flag_aero)
    gui.dt_aero.setEnabled(flag_aero)
    gui.label_dt_aero.setEnabled(flag_aero)
    gui.dt_uhma.setEnabled(flag_aero)
    gui.label_dt_uhma.setEnabled(flag_aero)
    gui.aero_start_label.setEnabled(flag_aero)
    gui.aero_start_offset.setEnabled(flag_aero)
    gui.aero_start_date.setEnabled(flag_aero)

    flag_mix_aero = flag.get("flag_mix_aero", 1) != 0
    gui.flag_mix_aero.setChecked(flag_mix_aero)

    aer_nucleation = aer.get("nucleation", True)
    gui.aer_nucleation.setChecked(aer_nucleation)

    aer_condensation = aer.get("condensation", True)
    gui.aer_condensation.setChecked(aer_condensation)

    aer_coagulation = aer.get("coagulation", True)
    gui.aer_coagulation.setChecked(aer_coagulation)

    aer_dry_deposition = aer.get("dry_deposition")
    gui.aer_dry_deposition.setChecked(aer_dry_deposition)

    # Note: Aerosol wet deposition is not yet implemented
    gui.aer_wet_deposition.setChecked(False)

    aer_snow_scavenge = aer.get("snow_scavenge")
    gui.aer_snow_scavenge.setChecked(aer_snow_scavenge)

    flag_station_model = flag.get("flag_model_type", 1) == 1
    gui.flag_station_model.setChecked(flag_station_model)
    gui.flag_trajectory_model.setChecked(not flag_station_model)
    gui.group_trajectory_model.setEnabled(not flag_station_model)

    flag_vapor = flag.get("flag_vapor", 0) != 0
    gui.flag_vapor.setChecked(flag_vapor)

    flag_debug = flag.get("flag_debug", 0) != 0
    gui.flag_debug.setChecked(flag_debug)


def _update_gui_from_grid_settings(gui):
    grid = _settings.get("nml_grid", dict())

    gui.masl.setValue(float(grid.get("masl", 1800.0)))
    gui.lat_deg.setValue(float(grid.get("lat_deg", 61.85)))
    gui.lon_deg.setValue(float(grid.get("lon_deg", 24.28)))


def _update_gui_from_time_settings(gui):
    time = _settings.get("nml_time", dict())

    gui.start_date.setDateTime(
        QtCore.QDateTime.fromString(
            ",".join(f"{s:02}" for s in time.get("start_date", [2000, 1, 1, 0, 0, 0])),
            "yyyy,MM,dd,HH,mm,ss",
        )
    )
    gui.end_date.setDateTime(
        QtCore.QDateTime.fromString(
            ",".join(f"{s:02}" for s in time.get("end_date", [2000, 1, 1, 0, 0, 0])),
            "yyyy,MM,dd,HH,mm,ss",
        )
    )

    # Note: A negative value would be misinterpreted afterwards
    gui.trajectory_duration.setValue(
        abs(gui.start_date.dateTime().daysTo(gui.end_date.dateTime()))
    )

    if gui.flag_trajectory_model.isChecked():
        fullDays = gui.trajectory_duration.value()

        if fullDays < 0:
            gui.start_date.setEnabled(False)
            gui.end_date.setEnabled(True)

            gui.start_date.setDate(gui.end_date.date().addDays(fullDays))
            gui.start_date.setTime(QtCore.QTime(0, 0))
        else:
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

    aero_start_date = QtCore.QDateTime.fromString(
        ",".join(f"{s:02}" for s in time.get("aero_start_date", [2000, 1, 1, 0, 0, 0])),
        "yyyy,MM,dd,HH,mm,ss",
    )
    secs_from_start = gui.start_date.dateTime().secsTo(aero_start_date)

    if secs_from_start < 0:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("Eager early aerosols")
        msg.setInformativeText(
            "The aerosol simulation cannot start before the simulation itself. Its starting time will be clamped to the simulation start."
        )
        msg.setWindowTitle("Warning loading INITFILE")
        msg.exec_()

    gui.aero_start_offset.setValue(secs_from_start // (60 * 60))
    gui.aero_start_date.setText(
        gui.start_date.dateTime()
        .addSecs(gui.aero_start_offset.value() * 60 * 60)
        .toString(" dd/MM/yyyy HH:mm:ss")
    )

    gui.dt_obs.setValue(float(time.get("dt_obs", 1800)))
    gui.dt_mete.setValue(float(time.get("dt_mete", 10)))
    gui.dt_emis.setValue(float(time.get("dt_emis", 60)))
    gui.dt_chem.setValue(float(time.get("dt_chem", 60)))
    gui.dt_depo.setValue(float(time.get("dt_depo", 60)))
    gui.dt_aero.setValue(float(time.get("dt_aero", 60)))
    gui.dt_uhma.setValue(float(time.get("dt_uhma", 10)))

    gui.time_zone.setValue(float(time.get("time_zone", 2.0)))

    if gui.time_zone.value() >= 0.0:
        gui.time_zone.setPrefix("UTC+")
    else:
        gui.time_zone.setPrefix("UTC")


def _update_gui_from_output_settings(gui):
    output = _settings.get("nml_output", dict())

    gui.output_list_spc.setPlainText(
        ", ".join(
            s.strip()
            for s in output.get("output_list_spc", "").split(",")
            if len(s.strip()) > 0
        )
    )
    gui.output_list_emi.setPlainText(
        ", ".join(
            s.strip()
            for s in output.get("output_list_emi", "").split(",")
            if len(s.strip()) > 0
        )
    )
    gui.output_list_Vd.setPlainText(
        ", ".join(
            s.strip()
            for s in output.get("output_list_vd", "").split(",")
            if len(s.strip()) > 0
        )
    )
    gui.output_list_vap.setPlainText(
        ", ".join(
            s.strip()
            for s in output.get("output_list_vap", "").split(",")
            if len(s.strip()) > 0
        )
    )

    gui.description.setPlainText(output.get("!description", ""))


def _update_gui_from_custom_settings(gui):
    custom = _settings.get("nml_custom", None)

    if custom is not None:
        i = 1

        for key, value in custom.items():
            getattr(gui, f"customKey_{i}").setText(str(key))
            getattr(gui, f"customVal_{i}").setText(str(value))

            i += 1

            if i > 30:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.setText("Excessive custom config")
                msg.setInformativeText(
                    "The INITFILE contains more than 30 properties in the "
                    + f"'NML_CUSTOM' namelist. Not all of the {len(custom)} "
                    + "properties are shown in the GUI."
                )
                msg.setWindowTitle("Warning loading INITFILE")
                msg.exec_()

                break

        for i in range(i, 31):
            getattr(gui, f"customKey_{i}").setText("")
            getattr(gui, f"customVal_{i}").setText("")
    else:
        for i in range(1, 31):
            getattr(gui, f"customKey_{i}").setText("")
            getattr(gui, f"customVal_{i}").setText("")


def _update_gui_from_raw_settings(gui, raw):
    gui.rawEdit.document().setPlainText(raw)


def _update_gui_from_settings(gui, raw):
    _update_gui_from_main_settings(gui)
    _update_gui_from_flag_settings(gui)
    _update_gui_from_grid_settings(gui)
    _update_gui_from_time_settings(gui)
    _update_gui_from_output_settings(gui)
    _update_gui_from_custom_settings(gui)
    _update_gui_from_raw_settings(gui, raw)


def load_settings(gui, path):
    with open(path, "r") as file:
        content = file.read()

    # Hack to read in the commented-out description
    content = content.replace("!DESCRIPTION", "DESCRIPTION", 1)

    # Hack to ensure that _settings is not mistaken as a local variable
    globals()["_settings"] = f90nml.reads(content)

    # Hack to ensure that the description is found as !description
    output = globals()["_settings"].get("nml_output")
    if output is not None:
        if output.get("description") is not None:
            output["!description"] = output.pop("description")

    _settings.indent = "  "
    _settings.end_comma = True
    _settings.uppercase = True

    raw = "\n".join(m.group(1) for m in _raw_setting_pattern.finditer(content))

    _update_gui_from_settings(gui, raw)


def _update_main_settings_from_gui(gui):
    _settings.patch(
        {
            "NML_MAIN": {
                "work_dir": str(Path(gui.main_dir.text()).resolve()),
                # Note: code_dir is replicated from work_dir
                "code_dir": str(Path(gui.main_dir.text()).resolve()),
                "case_dir": str(Path(gui.case_dir.text())),
                "chem_dir": str(Path(gui.chem_dir.text())),
                "input_dir": str(Path(gui.input_dir.text()).resolve()),
                "output_dir": str(Path(gui.output_dir.text())),
                "station": str(Path(gui.station.text())),
            }
        }
    )


def _update_flag_settings_from_gui(gui):
    _settings.patch(
        {
            "NML_FLAG": {
                "flag_emis": 2 if gui.flag_emis.isChecked() else 0,
                "flag_chem": 1 if gui.flag_chem.isChecked() else 0,
                "flag_gasdrydep": 1 if gui.flag_gasdrydep.isChecked() else 0,
                "flag_aero": 1 if gui.flag_aero.isChecked() else 0,
                # Note: Soil emissions are currently always disabled
                "flag_emis_soil": 0,
                "flag_debug": 1 if gui.flag_debug.isChecked() else 0,
                "flag_vapor": 1 if gui.flag_vapor.isChecked() else 0,
                # Note: The output lists are always read from the INITFILE
                "flag_outlist": 0,
                # Note: The format flag is always set to 1
                "flag_format": 1,
                "flag_model_type": 1 if gui.flag_station_model.isChecked() else 2,
                "flag_mix_chem": 1 if gui.flag_mix_chem.isChecked() else 0,
                "flag_aero": 1 if gui.flag_aero.isChecked() else 0,
                "flag_mix_aero": 1 if gui.flag_mix_aero.isChecked() else 0,
                # Note: The aerosols are always simulated in parallel
                "use_parallel_aerosol": True,
            },
            "AER_FLAG": {
                "options": {
                    "nucleation": gui.aer_nucleation.isChecked(),
                    "condensation": gui.aer_condensation.isChecked(),
                    "coagulation": gui.aer_coagulation.isChecked(),
                    "dry_deposition": gui.aer_dry_deposition.isChecked(),
                    # Note: Aerosol wet deposition is not yet implemented
                    # "wet_deposition": gui.aer_wet_deposition.isChecked(),
                    "snow_scavenge": gui.aer_snow_scavenge.isChecked(),
                },
            },
        }
    )


def _update_grid_settings_from_gui(gui):
    _settings.patch(
        {
            "NML_GRID": {
                "masl": gui.masl.value(),
                "lat_deg": gui.lat_deg.value(),
                "lon_deg": gui.lon_deg.value(),
            },
        }
    )


def _update_time_settings_from_gui(gui):
    _settings.patch(
        {
            "NML_TIME": {
                "start_date": [
                    int(i)
                    for i in gui.start_date.dateTime()
                    .toString("yyyy,MM,dd,HH,mm,ss")
                    .split(",")
                ],
                "end_date": [
                    int(i)
                    for i in gui.end_date.dateTime()
                    .toString("yyyy,MM,dd,HH,mm,ss")
                    .split(",")
                ],
                "aero_start_date": [
                    int(i)
                    for i in gui.start_date.dateTime()
                    .addSecs(gui.aero_start_offset.value() * (60 * 60))
                    .toString("yyyy,MM,dd,HH,mm,ss")
                    .split(",")
                ],
                "dt_obs": gui.dt_obs.value(),
                "dt_mete": gui.dt_mete.value(),
                "dt_emis": gui.dt_emis.value(),
                "dt_chem": gui.dt_chem.value(),
                "dt_depo": gui.dt_depo.value(),
                "dt_aero": gui.dt_aero.value(),
                "dt_uhma": gui.dt_uhma.value(),
                "time_zone": gui.time_zone.value(),
            },
        }
    )


def _update_output_settings_from_gui(gui):
    _settings.patch(
        {
            "NML_OUTPUT": {
                "output_list_spc": ", ".join(
                    s.strip()
                    for s in gui.output_list_spc.toPlainText().split(",")
                    if len(s.strip()) > 0
                ),
                "output_list_emi": ", ".join(
                    s.strip()
                    for s in gui.output_list_emi.toPlainText().split(",")
                    if len(s.strip()) > 0
                ),
                "output_list_Vd": ", ".join(
                    s.strip()
                    for s in gui.output_list_Vd.toPlainText().split(",")
                    if len(s.strip()) > 0
                ),
                "output_list_vap": ", ".join(
                    s.strip()
                    for s in gui.output_list_vap.toPlainText().split(",")
                    if len(s.strip()) > 0
                ),
                "!description": gui.description.toPlainText(),
            }
        }
    )


def _update_custom_settings_from_gui(gui):
    for i in range(1, 31):
        key = getattr(gui, f"customKey_{i}").text().strip()
        value = getattr(gui, f"customVal_{i}").text().strip()

        if key != "" and value != "":
            while True:
                # Integer value
                try:
                    value = int(value)
                    break
                except:
                    pass

                # Real value
                try:
                    value = float(value)
                    break
                except:
                    pass

                # Logical value
                if value.upper() == ".TRUE.":
                    value = True
                    break
                elif value.upper() == ".FALSE.":
                    value = False
                    break

                # Integer array
                try:
                    value = [int(v) for v in value.split(",")]
                    break
                except:
                    pass

                # Real array
                try:
                    value = [float(v) for v in value.split(",")]
                    break
                except:
                    pass

                # String value
                value = str(value)
                break

            _settings.patch(
                {
                    "NML_CUSTOM": {
                        key: value,
                    },
                }
            )


def _update_settings_from_gui(gui):
    _update_main_settings_from_gui(gui)
    _update_flag_settings_from_gui(gui)
    _update_grid_settings_from_gui(gui)
    _update_time_settings_from_gui(gui)
    _update_output_settings_from_gui(gui)
    _update_custom_settings_from_gui(gui)


def print_settings(gui):
    _update_settings_from_gui(gui)

    print("\n" * 10)
    print(f"! {'='*76} !")
    print(f"! {f'SOSAA setting file {_version_major}.{_version_minor}'.center(76)} !")
    print(
        f"! {('Created at: ' + time.strftime('%B %d %Y, %H:%M:%S', time.localtime())).center(76)} !"
    )
    print(f"! {'='*76} !")

    # Hack to ensure that _version_minor is not mistaken as a local variable
    globals()["_version_minor"] += 1

    print()
    print(_settings)

    print()
    print(_raw_settings_header)
    print(gui.rawEdit.toPlainText())
    print(_raw_settings_footer)


def write_settings(gui, path):
    _update_settings_from_gui(gui)

    with open(path, "w") as file:
        file.write(f"! {'='*76} !\n")
        file.write(
            f"! {f'SOSAA setting file {_version_major}.{_version_minor}'.center(76)} !\n"
        )
        file.write(
            f"! {('Created at: ' + time.strftime('%B %d %Y, %H:%M:%S', time.localtime())).center(76)} !\n"
        )
        file.write(f"! {'='*76} !\n")

        # Hack to ensure that _version_minor is not mistaken as a local variable
        globals()["_version_minor"] += 1

        file.write("\n")
        file.write(str(_settings))
        file.write("\n")

        file.write("\n")
        file.write(_raw_settings_header)
        file.write("\n")
        file.write(gui.rawEdit.toPlainText())
        file.write("\n")
        file.write(_raw_settings_footer)
        file.write("\n")
