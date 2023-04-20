import re
import sys
import time
from io import StringIO

import f90nml
from f90nml.namelist import Namelist

from ..config import get_config, set_config
from ..resources import resource_path
from .from_file import update_gui_from_settings
from .from_gui import update_settings_from_gui

default_settings_path = resource_path("conf/defaults.init")
minimal_settings_path = resource_path("conf/minimal.init")


def is_loading():
    return globals().get("_is_loading", False)


def load_settings(gui, path):
    globals()["_is_loading"] = True

    print(f"Loading INITFILE from {path} ...", flush=True)

    with open(path, "r") as file:
        content = file.read()

    # Hack to ensure that _settings is not mistaken as a local variable
    globals()["_settings"] = f90nml.reads(
        _raw_setting_pattern.sub("", content)
    )

    # Configure the f90 namelist pretty-printing options
    _settings.indent = "  "
    _settings.end_comma = True
    _settings.uppercase = True

    # Extract and combine the raw inputs
    raw = "\n".join(m.group(1) for m in _raw_setting_pattern.finditer(content))

    update_gui_from_settings(_settings, gui, raw)

    display_settings(gui)

    globals()["_is_loading"] = False


def display_settings(gui):
    config = StringIO()

    _write_settings_to_file(gui, config, update_version=False)

    gui.currentInitFileContent.document().setPlainText(config.getvalue())


def print_settings(gui):
    print("\n" * 10, flush=True)

    _write_settings_to_file(gui, sys.stdout, update_version=True)
    sys.stdout.flush()

    display_settings(gui)


def save_settings(gui, path):
    print(f"Saving INITFILE to {path} ...", flush=True)

    with open(path, "w") as file:
        _write_settings_to_file(gui, file, update_version=True)

    display_settings(gui)


def _write_settings_to_file(gui, file, update_version=True):
    update_settings_from_gui(_settings, gui)

    # fmt: off
    file.write(f"! {'='*76} !\n")
    file.write(
        f"! {f'SOSAA INITFILE {_version_major}.{_version_minor}'.center(76)} !\n"  # noqa: E501
    )
    file.write(
        f"! {('Created at: ' + time.strftime('%B %d %Y, %H:%M:%S', time.localtime())).center(76)} !\n"  # noqa: E501
    )
    file.write(f"! {'='*76} !\n")
    # fmt: on

    if update_version:
        # Hack to ensure that _version_minor is not mistaken
        #  as a local variable
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
_raw_settings_header = (
    f"! \\/ {'Raw input from the SOSAA GUI'.center(70, '-')} \\/ !"
)
_raw_settings_footer = (
    f"! /\\ {'Raw input from the SOSAA GUI'.center(70, '-')} /\\ !"
)
# fmt: off
_raw_setting_pattern = re.compile(
    rf"^{re.escape(_raw_settings_header)}$\n(.*?)\n^{re.escape(_raw_settings_footer)}$",  # noqa: E501
    flags=(re.MULTILINE | re.DOTALL),
)
# fmt: on
