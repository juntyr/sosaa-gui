# GUI for configuring the SOSAA model

## Installation

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
