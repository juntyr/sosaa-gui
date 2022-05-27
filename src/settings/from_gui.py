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


def update_settings_from_gui(settings, gui):
    _update_main_settings_from_gui(settings, gui)
    _update_flag_settings_from_gui(settings, gui)
    _update_grid_settings_from_gui(settings, gui)
    _update_time_settings_from_gui(settings, gui)
    _update_output_settings_from_gui(settings, gui)
    _update_custom_settings_from_gui(settings, gui)


def _update_main_settings_from_gui(settings, gui):
    settings.patch(
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


def _update_flag_settings_from_gui(settings, gui):
    settings.patch(
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


def _update_grid_settings_from_gui(settings, gui):
    settings.patch(
        {
            "NML_GRID": {
                "masl": gui.masl.value(),
                "lat_deg": gui.lat_deg.value(),
                "lon_deg": gui.lon_deg.value(),
            },
        }
    )


def _update_time_settings_from_gui(settings, gui):
    settings.patch(
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
                # Note: aero_start_date = start_date + aero_start_offset
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


def _update_output_settings_from_gui(settings, gui):
    settings.patch(
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


def _update_custom_settings_from_gui(settings, gui):
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

            settings.patch(
                {
                    "NML_CUSTOM": {
                        key: value,
                    },
                }
            )
