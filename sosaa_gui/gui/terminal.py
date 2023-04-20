import os
from shutil import which

from PyQt5 import QtWidgets


def generateTerminalCommand(command: str, frameId):
    if which("urxvt") is not None:
        term = "urxvt"
        embed = "-embed"
    elif which("rxvt") is not None:
        term = "rxvt"
        embed = "-embed"
    elif which("uxterm") is not None:
        term = "uxterm"
        embed = "-into"
    elif which("xterm") is not None:
        term = "xterm"
        embed = "-into"
    else:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Missing terminal")
        msg.setInformativeText(
            "Please install any of xterm, uxterm, rxvt, or urxvt."
        )
        msg.setWindowTitle("Error launching terminal")
        msg.exec_()

        return

    return term, [
        embed,
        f"{frameId}",
        "-sb",
        "-geometry",
        "640x480",
        "-hold",
        "-e",
        os.environ.get("SHELL", "sh"),
        "-x",
        "-c",
        command,
    ]
