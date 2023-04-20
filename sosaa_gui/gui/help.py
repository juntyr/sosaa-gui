from PyQt5 import QtCore, QtGui, QtWidgets

from ..layouts import about
from ..resources import resource_path


class About(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(About, self).__init__(parent)
        self.ab = about.Ui_Dialog()
        self.ab.setupUi(self)
        self.ab.okgreat.clicked.connect(self.reject)
        self.setWindowIcon(QtGui.QIcon(resource_path("icons/thebox_ico.png")))
        self.ab.logo.setPixmap(QtGui.QPixmap(resource_path("icons/ArcaLogoHR.png")))


def init_gui_help(gui):
    def openAboutHelp():
        About().exec()

    gui.actionAbout_SOSAA.triggered.connect(openAboutHelp)

    def openSOSAAWebpage():
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl(
                "https://www.helsinki.fi/en/researchgroups/multi-scale-modelling/sosaa"
            )
        )

    gui.actionSOSAA_webpage.triggered.connect(openSOSAAWebpage)

    def openSOSAAManual():
        return

    gui.actionOnline_manual.triggered.connect(openSOSAAManual)
