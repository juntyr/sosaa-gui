# SOSAA GUI &emsp; [![PyPi]][pypi-url] [![License]][gpl-3.0] [![CI Status]][ci-status]

[License]: https://img.shields.io/badge/License-GPL--3.0-blue.svg
[gpl-3.0]: https://www.gnu.org/licenses/gpl-3.0.html

[PyPI]: https://img.shields.io/pypi/v/sosaa-gui
[pypi-url]: https://pypi.org/project/sosaa-gui

[CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/sosaa-gui/ci.yml?branch=main&label=CI
[ci-status]: https://github.com/juntyr/sosaa-gui/actions/workflows/ci.yml?query=branch%3Amain

The SOSAA GUI provides a graphical user interface to configure, compile, run, and explore the [SOSAA model](https://www.helsinki.fi/en/researchgroups/multi-scale-modelling/sosaa). In particular, it allows the user to load, edit, and save INITFILEs for the model. The GUI is fully independent of SOSAA and can thus be run on a separate machine if needed. However, if SOSAA is installed on the same machine, the GUI can also be used to compile and run the model.

## Installation

### pip

The SOSAA GUI is available on the Python Package Index (PyPI) and can be installed using
```bash
pip install sosaa-gui
```
This command can also be run inside a conda environment to install the SOSAA GUI with conda.

### From Source

First, clone the git repository using
```bash
git clone https://github.com/juntyr/sosaa-gui.git
```
or
```bash
git clone git@github.com:juntyr/sosaa-gui.git
```

Next, enter the repository folder and use `pip` to install the program:
```bash
cd sosaa-gui && pip install .
```

Finally, you can launch the GUI using
```bash
sosaa-gui
```

## SOSAA Model Availability

Unfortunately, the SOSAA model is not yet publicly available. However, access to the complete SOSAA source code is provided upon request -- please contact Michael Boy (michael.boy@helsinki.fi), Putian Zhou (putian.zhou@helsinki.fi), or Petri Clusius (petri.clusius@helsinki.fi) for more information.

## Usage Notes

If you experience problems with the UI scaling, try launching the GUI as follows:
```bash
QT_SCREEN_SCALE_FACTORS="1.17;1.17" sosaa-gui
```

Note that you will need `xterm`, `uxterm`, `rxvt`, or `urxvt` installed on your system and in your path if you want to compile or run SOSAA *from within* the SOSAA GUI.

## License

Licensed under the GPL-3.0 license ([LICENSE-GPL](LICENSE-GPL) or https://www.gnu.org/licenses/gpl-3.0.html).

## Citation

Please refer to the [CITATION.cff](CITATION.cff) file and refer to https://citation-file-format.github.io to extract the citation in a format of your choice.

The SOSAA GUI was created as part of [Juniper Tyree](https://github.com/juntyr)'s Masters Thesis ["Prudent Response Surface Models"](https://github.com/juntyr/prudent-response-surface-models) for the M.Sc. Theoretical and Computational Methods programme at the University of Helsinki.
