from PyQt5 import QtGui, QtCore


def init_gui_help(gui):
    def openAboutHelp():
        return  # About().exec()

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
