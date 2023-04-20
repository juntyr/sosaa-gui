import colorsys

from PyQt5.QtGui import QColor, QPalette


# Convert a HSL colour to an RNG hex string
def hsl_to_hex(hue, saturation, lightness):
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation / 100, lightness / 100)

    return f"#{round(r*255):02X}{round(g*255):02X}{round(b*255):02X}"


# Generate a system of colours from a colour hue
def _coloursystem_from_hue(hue):
    return {
        "G0": hsl_to_hex(hue, 0, 0),
        "G10": hsl_to_hex(hue, 29, 14),
        "G20": hsl_to_hex(hue, 25, 21),
        "G30": hsl_to_hex(hue, 18, 26),
        "G40": hsl_to_hex(hue, 18, 33),
        "G50": hsl_to_hex(hue, 18, 40),
        "G60": hsl_to_hex(hue, 18, 46),
        "G70": hsl_to_hex(hue, 15, 54),
        "G80": hsl_to_hex(hue, 14, 66),
        "G90": hsl_to_hex(hue, 6, 69),
        "G100": hsl_to_hex(hue, 6, 74),
        "G110": hsl_to_hex(hue, 7, 80),
        "G120": hsl_to_hex(hue, 7, 82),
        "G130": hsl_to_hex(hue, 5, 88),
        "G140": hsl_to_hex(hue, 0, 98),
        "G150": hsl_to_hex(hue, 0, 100),
        "C0": hsl_to_hex(hue, 100, 0),
        "C10": hsl_to_hex(hue, 84, 15),
        "C20": hsl_to_hex(hue, 48, 28),
        "C30": hsl_to_hex(hue, 40, 36),
        "C40": hsl_to_hex(hue, 47, 39),
        "C50": hsl_to_hex(hue, 76, 42),
        "C60": hsl_to_hex(hue, 95, 41),
        "C70": hsl_to_hex(hue, 82, 53),
        "C80": hsl_to_hex(hue, 99, 61),
        "C90": hsl_to_hex(hue, 100, 73),
        "C100": hsl_to_hex(hue, 100, 81),
        "C110": hsl_to_hex(hue, 85, 87),
        "C120": hsl_to_hex(hue, 100, 90),
        "C130": hsl_to_hex(hue, 100, 93),
        "C140": hsl_to_hex(hue, 100, 98),
        "C150": hsl_to_hex(hue, 100, 100),
    }


# Generate a light/dark QPalette from a colour hue
def get_style_palette(hue, dark):
    coloursystem = _coloursystem_from_hue(hue)

    palette = QPalette()

    if dark:
        palette.setColor(QPalette.Window, QColor(coloursystem["G30"]))
        palette.setColor(QPalette.WindowText, QColor(coloursystem["G150"]))
        palette.setColor(QPalette.Base, QColor(coloursystem["G20"]))
        palette.setColor(QPalette.AlternateBase, QColor(coloursystem["G30"]))
        palette.setColor(QPalette.ToolTipBase, QColor(coloursystem["G10"]))
        palette.setColor(QPalette.ToolTipText, QColor(coloursystem["G150"]))
        palette.setColor(QPalette.Text, QColor(coloursystem["G150"]))
        palette.setColor(QPalette.Button, QColor(coloursystem["G30"]))
        palette.setColor(QPalette.ButtonText, QColor(coloursystem["G150"]))
        palette.setColor(QPalette.BrightText, QColor(coloursystem["C150"]))
        palette.setColor(QPalette.Link, QColor(coloursystem["C70"]))
        palette.setColor(QPalette.Highlight, QColor(coloursystem["G120"]))
        palette.setColor(QPalette.HighlightedText, QColor(coloursystem["G20"]))
        palette.setColor(
            QPalette.Active, QPalette.Button, QColor(coloursystem["G50"])
        )
        palette.setColor(
            QPalette.Disabled,
            QPalette.ButtonText,
            QColor(coloursystem["G100"]),
        )
        palette.setColor(
            QPalette.Disabled,
            QPalette.WindowText,
            QColor(coloursystem["G100"]),
        )
        palette.setColor(
            QPalette.Disabled, QPalette.Text, QColor(coloursystem["G100"])
        )
        palette.setColor(
            QPalette.Disabled, QPalette.Light, QColor(coloursystem["G30"])
        )
    else:
        palette.setColor(QPalette.Window, QColor(coloursystem["G120"]))
        palette.setColor(QPalette.WindowText, QColor(coloursystem["G0"]))
        palette.setColor(QPalette.Base, QColor(coloursystem["G130"]))
        palette.setColor(QPalette.AlternateBase, QColor(coloursystem["G120"]))
        palette.setColor(QPalette.ToolTipBase, QColor(coloursystem["G140"]))
        palette.setColor(QPalette.ToolTipText, QColor(coloursystem["G0"]))
        palette.setColor(QPalette.Text, QColor(coloursystem["G0"]))
        palette.setColor(QPalette.Button, QColor(coloursystem["G120"]))
        palette.setColor(QPalette.ButtonText, QColor(coloursystem["G0"]))
        palette.setColor(QPalette.BrightText, QColor(coloursystem["C10"]))
        palette.setColor(QPalette.Link, QColor(coloursystem["C80"]))
        palette.setColor(QPalette.Highlight, QColor(coloursystem["C30"]))
        palette.setColor(
            QPalette.HighlightedText, QColor(coloursystem["G130"])
        )
        palette.setColor(
            QPalette.Active, QPalette.Button, QColor(coloursystem["G100"])
        )
        palette.setColor(
            QPalette.Disabled, QPalette.ButtonText, QColor(coloursystem["G50"])
        )
        palette.setColor(
            QPalette.Disabled, QPalette.WindowText, QColor(coloursystem["G50"])
        )
        palette.setColor(
            QPalette.Disabled, QPalette.Text, QColor(coloursystem["G50"])
        )
        palette.setColor(
            QPalette.Disabled, QPalette.Light, QColor(coloursystem["G120"])
        )

    return palette
