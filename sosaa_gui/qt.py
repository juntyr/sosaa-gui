import os
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication

try:
    import platform

    # "Windows" / "Linux" / "Darwin"
    operating_system = platform.system() or "Linux"
except Exception:
    operating_system = "Linux"


# Set the GUI scaling factor for QT based on the operating system
def setup_qt_scaling():
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    # Enable QT highdpi scaling and highdpi icons
    QtWidgets.QApplication.setAttribute(
        QtCore.Qt.AA_EnableHighDpiScaling, True
    )
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # Check if scaling is necessary, currently only required on Windows
    args = []

    for arg in sys.argv:
        args.append(arg.upper())

        if "--scaling_" in arg:
            os.environ["QT_SCALE_FACTOR"] = (
                f"{float(arg.replace('--scaling_', '')):3.2f}"
            )

            args.append("-NS")

    # FIXME petri: check if platform usually works for everyone
    if (
        os.name.upper() == "NT" or operating_system == "Windows"
    ) and "-NS" not in args:
        try:
            import ctypes

            sf = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
            if sf < 1.3:
                sf = 1
            sf = 1 / sf

            u32 = ctypes.windll.user32
            scrhgt = u32.GetSystemMetrics(1)

            if scrhgt < 850:
                sf = sf * scrhgt / 850.0
        except Exception:
            sf = 1

            print(
                (
                    "Failed to get the scaling factor of the screen, falling"
                    f" back to {sf:3.2f}."
                ),
                flush=True,
            )

        os.environ["QT_SCALE_FACTOR"] = f"{sf:3.2f}"

    if "-NS" in args:
        print(
            (
                "Scaling factor overriden to"
                f" {os.environ['QT_SCALE_FACTOR']} from the commandline."
            ),
            flush=True,
        )


# Set the QT style for the GUI application
def setup_qt_style():
    styles = QtWidgets.QStyleFactory.keys()

    if "Fusion" in styles:
        QCoreApplication.instance().setStyle(
            QtWidgets.QStyleFactory.create("Fusion")
        )
    else:
        print(f"Available styles: {styles}", flush=True)
