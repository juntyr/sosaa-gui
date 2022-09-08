import os

import darkdetect

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QCoreApplication

from ..config import get_config, set_config, remove_config
from ..resources import resource_path
from ..style import hsl_to_hex, get_style_palette


def init_gui_style(gui):
    _refresh_style(gui)

    # Style can be EITHER system, dark, or light
    gui.themeGroup = QtWidgets.QActionGroup(gui)
    gui.themeGroup.addAction(gui.actionSystem)
    gui.themeGroup.addAction(gui.actionDark)
    gui.themeGroup.addAction(gui.actionLight)

    # Open a colour picker window to select the new style colour
    def actionStyleColourTrigger():
        colourSelector = QtWidgets.QColorDialog(gui)
        colourSelector.setOption(
            QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog
        )
        colourSelector.setOption(QtWidgets.QColorDialog.ColorDialogOption.NoButtons)

        def colourChanged(colour):
            gui.hue = colour.getHsl()[0]
            set_config("style", "hue", str(gui.hue))
            _setLightDarkStyle(gui, gui.dark)

        colourSelector.setCurrentColor(QtGui.QColor(hsl_to_hex(gui.hue, 100, 100)))
        colourSelector.currentColorChanged.connect(colourChanged)

        colourSelector.exec_()

    gui.actionChange_Colour.triggered.connect(actionStyleColourTrigger)

    _loadDefaultStyle(gui)

    def actionSystemTrigger(checked):
        if checked:
            set_config("style", "theme", "system")
            _setLightDarkStyle(gui, darkdetect.isDark())

    gui.actionSystem.triggered.connect(actionSystemTrigger)

    def actionLightTrigger(checked):
        if checked:
            set_config("style", "theme", "light")
            _setLightDarkStyle(gui, False)

    gui.actionLight.triggered.connect(actionLightTrigger)

    def actionDarkTrigger(checked):
        if checked:
            set_config("style", "theme", "dark")
            _setLightDarkStyle(gui, True)

    gui.actionDark.triggered.connect(actionDarkTrigger)

    def actionResetStyleTrigger():
        remove_config("style", "hue")
        remove_config("style", "theme")

        _loadDefaultStyle(gui)

    gui.actionReset_Style.triggered.connect(actionResetStyleTrigger)

    _loadDefaultFont(gui)

    def actionSetGlobalFontTrigger():
        fontSelector = QtWidgets.QFontDialog()
        font, ok = fontSelector.getFont(gui.font(), parent=gui)

        if ok:
            set_config("style", "font", font.toString())
            _refresh_font(gui, font)

    gui.actionSet_Global_Font.triggered.connect(actionSetGlobalFontTrigger)

    def actionResetFontTrigger():
        remove_config("style", "font")

        _loadDefaultFont(gui)

    gui.actionReset_Fonts.triggered.connect(actionResetFontTrigger)


def _loadDefaultStyle(gui):
    gui.hue = int(get_config("style", "hue", fallback="316"))

    theme = get_config("style", "theme", fallback="system")

    if theme == "light":
        gui.actionLight.setChecked(True)
        _setLightDarkStyle(gui, False)
    elif theme == "dark":
        gui.actionDark.setChecked(True)
        _setLightDarkStyle(gui, True)
    else:
        gui.actionSystem.setChecked(True)
        _setLightDarkStyle(gui, darkdetect.isDark())


def _loadDefaultFont(gui):
    font_str = get_config("style", "font", fallback=None)

    if font_str is None:
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
    else:
        font = QtGui.QFont()
        font.fromString(font_str)

    _refresh_font(gui, font)


def _setLightDarkStyle(gui, dark):
    gui.dark = dark

    QCoreApplication.instance().setPalette(get_style_palette(gui.hue, gui.dark))

    _refresh_style(gui)


def _refresh_style(gui):
    gui.saveCurrentButton.setStyleSheet(
        buttonStyle(
            "icons/saveia.png"
            if gui.currentInitFileToSave is None
            else "icons/save.png"
        )
    )
    gui.saveButton.setStyleSheet(buttonStyle("icons/saveas.png"))
    gui.loadButton.setStyleSheet(buttonStyle("icons/load.png"))
    gui.saveDefaults.setStyleSheet(buttonStyle("icons/pack.png"))
    gui.recompile.setStyleSheet(buttonStyle("icons/recompile.png"))


def buttonStyle(icon):
    icon_path_escaped = resource_path(icon).replace("\\", "\\\\")

    return (
        f"background-image: url('{icon_path_escaped}');\nbackground-repeat: no-repeat;"
    )


def _refresh_font(gui, font):
    gui.setFont(font)

    gui.menuFile.setFont(font)
    gui.menuTools.setFont(font)
    gui.menuSettings.setFont(font)
    gui.menuStyle.setFont(font)
    gui.menuFont.setFont(font)
    gui.menuHelp.setFont(font)
