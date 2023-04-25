import darkdetect
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication

from ..config import get_config, remove_config, set_config
from ..resources import resource_path
from ..style import get_style_palette, hsl_to_hex
from .syntax import STYLES


def init_gui_style(gui):
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
        colourSelector.setOption(
            QtWidgets.QColorDialog.ColorDialogOption.NoButtons
        )

        def colourChanged(colour):
            gui.hue = colour.getHsl()[0]
            set_config("style", "hue", str(gui.hue))
            _setLightDarkStyle(gui, gui.dark)

        colourSelector.setCurrentColor(
            QtGui.QColor(hsl_to_hex(gui.hue, 100, 100))
        )
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

    QCoreApplication.instance().setPalette(
        get_style_palette(gui.hue, gui.dark)
    )

    _refresh_style(gui)


def _refresh_style(gui):
    gui.saveCurrentButton.setStyleSheet(
        buttonStyle(
            "icons/save-inactive.svg"
            if gui.currentInitFileToSave is None
            else "icons/save.svg"
        )
    )
    gui.saveButton.setStyleSheet(buttonStyle("icons/save-as.svg"))
    gui.loadButton.setStyleSheet(buttonStyle("icons/load.svg"))
    gui.saveDefaults.setStyleSheet(buttonStyle("icons/save-defaults.svg"))
    gui.recompile.setStyleSheet(buttonStyle("icons/recompile.svg"))

    gui.rsm_perturbation_header.setText(
        QtCore.QCoreApplication.translate(
            "MainWindow",
            (
                '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN"'
                ' "http://www.w3.org/TR/REC-html40/strict.dtd">\n'
                '<html><head><meta name="qrichtext" content="1" /><meta'
                ' charset="utf-8" /><style type="text/css">\n'
                "p, li { white-space: pre-wrap; }\n"
                "</style></head><body style=\" font-family:'Segoe UI';"
                ' font-size:9pt; font-weight:400; font-style:normal;">\n'
                '<p style=" margin-top:0px; margin-bottom:0px;'
                " margin-left:0px; margin-right:0px; -qt-block-indent:0;"
                ' text-indent:0px;"><span style=" font-weight:700;'
                f' color:\'{STYLES["keyword"][2+gui.dark]}\';">def</span>'
                ' <span style=" font-weight:700;'
                f' color:\'{STYLES["defclass"][2+gui.dark]}\';">'
                'perturb_inputs</span><span style=" font-weight:700;'
                f' color:\'{STYLES["brace"][2+gui.dark]}\';">(</span>inputs:'
                ' pandas<span style=" font-weight:700;'
                f' color:\'{STYLES["operator"][2+gui.dark]}\';">.</span>'
                'DataFrame<span style=" font-weight:700;'
                f' color:\'{STYLES["brace"][2+gui.dark]}\';">)</span> <span'
                ' style=" font-weight:700;'
                f' color:\'{STYLES["operator"][2+gui.dark]}\';">-&gt;</span>'
                ' pandas<span style=" font-weight:700;'
                f' color:\'{STYLES["operator"][2+gui.dark]}\';">.</span>'
                "DataFrame:</p></body></html>"
            ),
        )
    )

    if getattr(gui, "rawEditHighlight", None) is not None:
        gui.rawEditHighlight.rehighlight()
        gui.currentInitFileContentHighlight.rehighlight()
        gui.rsm_perturbation_highlight.rehighlight()


def buttonStyle(icon):
    icon_path_escaped = resource_path(icon).replace("\\", "\\\\")

    return (
        f"background-image: url('{icon_path_escaped}');\nbackground-repeat:"
        " no-repeat;"
    )


def _refresh_font(gui, font):
    gui.setFont(font)

    gui.menuFile.setFont(font)
    gui.menuTools.setFont(font)
    gui.menuSettings.setFont(font)
    gui.menuStyle.setFont(font)
    gui.menuFont.setFont(font)
    gui.menuHelp.setFont(font)

    monospace = QtGui.QFont()
    monospace.fromString(font.toString())
    monospace.setFamily("Courier New")
    monospace.setStyleHint(QtGui.QFont.StyleHint.Monospace)

    gui.rawEdit.setFont(monospace)
    gui.currentInitFileContent.setFont(monospace)
    gui.rsm_perturbation.setFont(monospace)
