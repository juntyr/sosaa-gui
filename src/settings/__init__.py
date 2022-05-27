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

import re
import time

import f90nml

from f90nml.namelist import Namelist

from ..config import get_config, set_config
from ..resources import resource_path
from .from_gui import update_settings_from_gui
from .from_file import update_gui_from_settings


default_settings_path = resource_path("conf/defaults.init")
minimal_settings_path = resource_path("conf/minimal.init")


def load_settings(gui, path):
    print(f"Loading INITFILE from {path} ...")

    with open(path, "r") as file:
        content = file.read()

    # Hack to read in the commented-out description
    content = content.replace("!DESCRIPTION", "DESCRIPTION", 1)

    # Hack to ensure that _settings is not mistaken as a local variable
    globals()["_settings"] = f90nml.reads(content)

    # Configure the f90 namelist pretty-printing options
    _settings.indent = "  "
    _settings.end_comma = True
    _settings.uppercase = True

    # Hack to ensure that the description is found as !description
    output = globals()["_settings"].get("nml_output")
    if output is not None:
        if output.get("description") is not None:
            output["!description"] = output.pop("description")

    # Extract and combine the raw inputs
    raw = "\n".join(m.group(1) for m in _raw_setting_pattern.finditer(content))

    update_gui_from_settings(_settings, gui, raw)


def print_settings(gui):
    update_settings_from_gui(_settings, gui)

    print("\n" * 10)
    print(f"! {'='*76} !")
    print(f"! {f'SOSAA INITFILE {_version_major}.{_version_minor}'.center(76)} !")
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


def save_settings(gui, path):
    print(f"Saving INITFILE to {path} ...")

    update_settings_from_gui(_settings, gui)

    with open(path, "w") as file:
        file.write(f"! {'='*76} !\n")
        file.write(
            f"! {f'SOSAA INITFILE {_version_major}.{_version_minor}'.center(76)} !\n"
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


# Create the module-local settings global and its pretty-printing options
_settings = Namelist()
_settings.indent = "  "
_settings.end_comma = True
_settings.uppercase = True


# FIXME petri: only change the minor version when the configuration changes
_version_major = int(get_config("settings", "version", fallback="0")) + 1
set_config("settings", "version", str(_version_major))
_version_minor = 0


# Regex pattern to insert and extract the raw input from the GUI
_raw_settings_header = f"! \\/ {'Raw input from the SOSAA GUI'.center(70, '-')} \\/ !"
_raw_settings_footer = f"! /\\ {'Raw input from the SOSAA GUI'.center(70, '-')} /\\ !"
_raw_setting_pattern = re.compile(
    rf"^{re.escape(_raw_settings_header)}$\n(.*?)\n^{re.escape(_raw_settings_footer)}$",
    flags=(re.MULTILINE | re.DOTALL),
)
