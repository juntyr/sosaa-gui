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
    print(f"Loading INITFILE from {path} ...", flush=True)

    with open(path, "r") as file:
        content = file.read()

    # Hack to read in commented-out variables
    content = re.sub(
        "!CHEMALL_DIR", "CHEMALL_DIR", content, count=1, flags=re.IGNORECASE
    )
    content = re.sub(
        "!CHEMNAME_DIR", "CHEMNAME_DIR", content, count=1, flags=re.IGNORECASE
    )
    content = re.sub(
        "!CASENAME_DIR", "CASENAME_DIR", content, count=1, flags=re.IGNORECASE
    )
    content = re.sub(
        "!DESCRIPTION", "DESCRIPTION", content, count=1, flags=re.IGNORECASE
    )

    # Hack to ensure that _settings is not mistaken as a local variable
    globals()["_settings"] = f90nml.reads(content)

    # Configure the f90 namelist pretty-printing options
    _settings.indent = "  "
    _settings.end_comma = True
    _settings.uppercase = True

    # Hack to ensure that chemall_dir, chemname_dir, and casename_dir
    #  are all found under their commented-out named
    main = globals()["_settings"].get("nml_main")
    if main is not None:
        if main.get("chemall_dir") is not None:
            main["!chemall_dir"] = main.pop("chemall_dir")
        if main.get("chemname_dir") is not None:
            main["!chemname_dir"] = main.pop("chemname_dir")
        if main.get("casename_dir") is not None:
            main["!casename_dir"] = main.pop("casename_dir")

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

    print("\n" * 10, flush=True)
    print(f"! {'='*76} !", flush=True)
    print(
        f"! {f'SOSAA INITFILE {_version_major}.{_version_minor}'.center(76)} !",
        flush=True,
    )
    print(
        f"! {('Created at: ' + time.strftime('%B %d %Y, %H:%M:%S', time.localtime())).center(76)} !",
        flush=True,
    )
    print(f"! {'='*76} !", flush=True)

    # Hack to ensure that _version_minor is not mistaken as a local variable
    globals()["_version_minor"] += 1

    print(flush=True)
    print(_settings, flush=True)

    print(flush=True)
    print(_raw_settings_header, flush=True)
    print(gui.rawEdit.toPlainText(), flush=True)
    print(_raw_settings_footer, flush=True)


def save_settings(gui, path):
    print(f"Saving INITFILE to {path} ...", flush=True)

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
