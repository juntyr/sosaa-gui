from PyQt5 import QtWidgets, QtGui, QtCore

from ..layouts import gui
from ..resources import resource_path
from ..version import sosaa_version_pretty
from .style import init_gui_style
from .loadsave import init_gui_loadsave
from .dirs import init_dirs_gui
from .modules import init_modules_gui
from .output import init_gui_output
from .scenario import init_scenario_gui
from .compile import init_compile_gui
from .help import init_gui_help


class QtSosaaGui(gui.Ui_MainWindow, QtWidgets.QMainWindow):
    """Main program window."""

    def __init__(self):
        super(QtSosaaGui, self).__init__()
        self.setupUi(self)

        self.currentInitFileToSave = None

        self.setWindowTitle(sosaa_version_pretty)
        self.setWindowIcon(QtGui.QIcon(resource_path("icons/thebox_ico.png")))

        self.actionQuit_Ctrl_Q.triggered.connect(self.close)

        init_gui_style(self)
        init_gui_loadsave(self)
        init_gui_output(self)
        init_gui_help(self)

        init_dirs_gui(self)
        init_modules_gui(self)
        init_scenario_gui(self)
        init_compile_gui(self)
