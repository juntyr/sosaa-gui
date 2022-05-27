"""
=============================================================================
Copyright (C) 2021  Multi-Scale Modelling group
Written by Zhou Putian and Juniper Langenstein, University of Helsinki
Institute for Atmospheric and Earth System Research (INAR), University of Helsinki
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


import sys
import os

if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Using SOSAA GUI requires Python 3.4 (preferrably 3.6 or later)", flush=True)
    quit()

csc = "--csc" in sys.argv

python = input('Type the command you use to call Python 3 (hit Enter for "python3"): ')

if python == "":
    python = "python3"

pyt = input("Install necessary Python packages? (y/n)?: ")

if pyt == "y" or pyt == "Y":
    user = input("Perform installation in user-space? (y/n)?: ")

    if user == "y" or user == "Y":
        user = "--user"
    else:
        user = ""

    outpyt = os.system(
        f"{python} -m pip install {user} PyQt5 f90nml darkdetect netCDF4"
    )

    if outpyt != 0:
        print(
            "Unfortunately the Python module installation did not work, updating setuptools could help.",
            flush=True,
        )

        upgr = input("Proceed and try again? (y/n)?: ")

        if upgr != "y" and upgr != "Y":
            quit()

        outpyt = os.system(f"{python} -m pip install --upgrade setuptools")
        outpyt += os.system(
            f"{python} -m pip install {user} PyQt5 f90nml darkdetect netCDF4"
        )

        if outpyt != 0:
            print(
                "Unfortunately still some Python modules failed to install.", flush=True
            )
            quit()

print(flush=True)
print("Congratulations! You can now run:", flush=True)
print(f"$ {python} SOSAA_gui.py", flush=True)
