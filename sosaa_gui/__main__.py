import signal
import time

from PyQt5 import QtWidgets

from sosaa_gui.gui import QtSosaaGui
from sosaa_gui.qt import setup_qt_scaling, setup_qt_style
from sosaa_gui.version import sosaa_version_pretty

# Enable GUI termination using ctrl-c
signal.signal(signal.SIGINT, signal.SIG_DFL)

print(
    (
        f"{sosaa_version_pretty} started at:"
        f" {time.strftime('%B %d %Y, %H:%M:%S', time.localtime())}"
    ),
    flush=True,
)

setup_qt_scaling()

app = QtWidgets.QApplication([])

gui = QtSosaaGui()
gui.setGeometry(30, 30, 900, 700)
gui.show()

setup_qt_style()

app.exec_()
