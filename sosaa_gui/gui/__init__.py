from PyQt5 import QtGui, QtWidgets

from ..layouts import gui
from ..resources import resource_path
from ..version import sosaa_version_pretty
from .compile import init_compile_gui
from .dirs import init_dirs_gui
from .help import init_gui_help
from .loadsave import init_gui_loadsave
from .modules import init_modules_gui
from .output import init_gui_output
from .rsm import init_rsm_gui
from .run import init_run_gui
from .scenario import init_scenario_gui
from .style import init_gui_style


class QtSosaaGui(gui.Ui_MainWindow, QtWidgets.QMainWindow):
    """Main program window."""

    def __init__(self):
        super(QtSosaaGui, self).__init__()
        self.setupUi(self)

        self.currentInitFileToSave = None

        self.setWindowTitle(sosaa_version_pretty)
        self.setWindowIcon(QtGui.QIcon(resource_path("icons/sosaa-icon.svg")))

        self.actionQuit_Ctrl_Q.triggered.connect(self.close)

        init_gui_style(self)
        init_gui_loadsave(self)
        init_gui_output(self)
        init_gui_help(self)

        init_dirs_gui(self)
        init_modules_gui(self)
        init_scenario_gui(self)
        init_compile_gui(self)
        init_run_gui(self)
        init_rsm_gui(self)
