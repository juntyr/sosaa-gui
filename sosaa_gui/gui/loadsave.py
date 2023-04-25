from pathlib import Path

from PyQt5 import QtWidgets

from ..settings import (
    default_settings_path,
    display_settings,
    load_settings,
    minimal_settings_path,
    print_settings,
    save_settings,
)
from ..version import sosaa_version_pretty
from .browse import browsePath
from .style import buttonStyle
from .syntax import FortranNamelistHighlighter


def init_gui_loadsave(gui):
    _hideCurrentInitFile(gui)

    gui.actionOpen.triggered.connect(lambda: _loadSettings(gui))
    gui.loadButton.clicked.connect(lambda: _loadSettings(gui))

    def loadMinimalSettings():
        gui.currentInitFileToSave = None
        load_settings(gui, minimal_settings_path)
        _hideCurrentInitFile(gui)

    gui.actionLoad_minimal_settings.triggered.connect(loadMinimalSettings)

    # If no default settings exist, just use the minimal ones
    if Path(default_settings_path).exists():
        load_settings(gui, default_settings_path)
    else:
        load_settings(gui, minimal_settings_path)

    gui.actionPrint.triggered.connect(lambda: print_settings(gui))

    gui.saveButton.clicked.connect(lambda: _saveSettings(gui, path=None))
    gui.actionSave_2.triggered.connect(lambda: _saveSettings(gui, path=None))

    gui.saveCurrentButton.clicked.connect(
        lambda: _saveSettings(gui, path=gui.currentInitFileToSave)
    )
    gui.actionSave_to_current.triggered.connect(
        lambda: _saveSettings(gui, path=gui.currentInitFileToSave)
    )

    gui.saveDefaults.clicked.connect(
        lambda: _saveSettings(gui, path=default_settings_path)
    )

    # Set up drag and drop to load init files
    type(gui).dragEnterEvent = _dragInitFileEnter
    type(gui).dropEvent = _dropInitFile
    gui.setAcceptDrops(True)

    # Set up raw INITFILE syntax highlighting
    gui.rawEditHighlight = FortranNamelistHighlighter(
        gui, gui.rawEdit.document()
    )

    # Set up INITFILE update on switch to its tab
    gui.tabWidget.currentChanged.connect(lambda i: _tab_switched(gui, i))

    # Set up INITFILE syntax highlighting
    gui.currentInitFileContentHighlight = FortranNamelistHighlighter(
        gui,
        gui.currentInitFileContent.document(),
    )


def _loadSettings(gui):
    path = browsePath(title="Choose INITFILE")

    _loadSettingsWithPath(gui, path)


def _loadSettingsWithPath(gui, path):
    if (
        str(path) != default_settings_path
        and str(path) != minimal_settings_path
    ):
        gui.currentInitFileToSave = path
    else:
        gui.currentInitFileToSave = None

    if path is not None:
        load_settings(gui, path)

    if gui.currentInitFileToSave is None:
        _hideCurrentInitFile(gui)
    else:
        _showCurrentInitFile(gui, gui.currentInitFileToSave)


def _saveSettings(gui, path=None):
    if path is None:
        path = browsePath(title="Save INITFILE", save=True)

    if path is None:
        return

    if str(path) == minimal_settings_path:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Invalid INITFILE path")
        msg.setInformativeText(
            "Overwriting the minimal INITFILE from the GUI is not allowed."
        )
        msg.setWindowTitle("Error saving INITFILE")
        msg.exec_()

        return

    save_settings(gui, path)

    if str(path) != default_settings_path:
        gui.currentInitFileToSave = path

        _showCurrentInitFile(gui, path)
    else:
        _hideCurrentInitFile(gui)


def _showCurrentInitFile(gui, path):
    gui.saveCurrentButton.setStyleSheet(buttonStyle("icons/save.svg"))
    gui.saveCurrentButton.setEnabled(True)
    gui.actionSave_to_current.setEnabled(True)

    pathstr = str(path)

    if str(path).startswith(str(Path.cwd())):
        path = path.relative_to(Path.cwd())
        pathstr = f"./{path}"

    gui.currentInitFile.setText(pathstr)

    gui.setWindowTitle(f"{sosaa_version_pretty}: {pathstr}")
    gui.saveCurrentButton.setToolTip(f"Save to {path.name}")


def _hideCurrentInitFile(gui):
    gui.saveCurrentButton.setEnabled(False)
    gui.actionSave_to_current.setEnabled(False)
    gui.saveCurrentButton.setStyleSheet(buttonStyle("icons/save-inactive.svg"))

    gui.currentInitFile.setText("")

    gui.setWindowTitle(sosaa_version_pretty)
    gui.saveCurrentButton.setToolTip("")


def _dragInitFileEnter(_gui, e):
    # Fake-access the gui variable
    _gui = _gui

    if e.mimeData().hasFormat("text/plain"):
        e.accept()
    else:
        e.ignore()


def _dropInitFile(gui, e):
    path = Path(e.mimeData().text().replace("file://", "").strip()).resolve()

    _loadSettingsWithPath(gui, path)


def _tab_switched(gui, _i):
    # Fake-access the i variable
    _i = _i

    if gui.tabWidget.currentWidget() == gui.initfile_tab:
        display_settings(gui)
