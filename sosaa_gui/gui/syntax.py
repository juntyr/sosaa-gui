"""
Python syntax highlighting adopted from:
https://wiki.python.org/moin/PyQt/Python%20syntax%20highlighting
Released under the BSD 3-Clause "New" or "Revised" License
"""

from PyQt5 import QtCore, QtGui


def _format(lightcolor, darkcolor, style=""):
    def _format_single(color, style):
        """Return a QTextCharFormat with the given attributes."""
        _color = QtGui.QColor()
        _color.setNamedColor(color)

        _format = QtGui.QTextCharFormat()
        _format.setForeground(_color)
        if "bold" in style:
            _format.setFontWeight(QtGui.QFont.Bold)
        if "italic" in style:
            _format.setFontItalic(True)
        if "underline" in style:
            _format.setFontUnderline(True)

        return _format

    return (
        _format_single(lightcolor, style),
        _format_single(darkcolor, style),
        lightcolor,
        darkcolor,
    )


# Syntax styles that can be shared by all languages
STYLES = {
    "keyword": _format("dodgerblue", "lightskyblue", "bold"),
    "operator": _format("orangered", "darkorange", "bold"),
    "brace": _format("slategray", "slategray", "bold"),
    "defclass": _format("indianred", "salmon", "bold"),
    "string": _format("green", "lawngreen"),
    "string2": _format("sienna", "sandybrown", "italic"),
    "comment": _format("sienna", "sandybrown", "italic"),
    "self": _format("mediumblue", "aqua", "underline"),
    "numbers": _format("royalblue", "deepskyblue", "bold"),
    "function": _format("mediumblue", "aqua"),
}


class PythonHighlighter(QtGui.QSyntaxHighlighter):
    """Syntax highlighter for the Python language."""

    # Python keywords
    keywords = [
        "and",
        "as",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "False",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "None",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "True",
        "try",
        "while",
        "with",
        "yield",
    ]

    # Python special functions
    functions = ["super", "len", "print"]

    # Python operators
    operators = [
        "=",
        # Comparison
        "==",
        "!=",
        "<",
        "<=",
        ">",
        ">=",
        # Arithmetic
        r"\+",
        "-",
        r"\*",
        "/",
        "//",
        r"\%",
        r"\*\*",
        # In-place
        r"\+=",
        "-=",
        r"\*=",
        r"/=",
        r"\%=",
        # Bitwise
        r"\^",
        r"\|",
        r"\&",
        r"\~",
        ">>",
        "<<",
        # Field access
        r"\.",
    ]

    # Python braces
    braces = [
        r"\{",
        r"\}",
        r"\(",
        r"\)",
        r"\[",
        r"\]",
    ]

    def __init__(self, gui, parent: QtGui.QTextDocument) -> None:
        super().__init__(parent)

        self.gui = gui

        # Multi-line strings (expression, flag, style)
        self.tri_single = (QtCore.QRegExp("'''"), 1, STYLES["string2"])
        self.tri_double = (QtCore.QRegExp('"""'), 2, STYLES["string2"])

        rules = []

        # Keyword, operator, brace, and special function rules
        rules += [
            (r"\b%s\b" % w, 0, STYLES["keyword"])
            for w in PythonHighlighter.keywords
        ]
        rules += [
            (r"%s" % o, 0, STYLES["operator"])
            for o in PythonHighlighter.operators
        ]
        rules += [
            (r"%s" % b, 0, STYLES["brace"]) for b in PythonHighlighter.braces
        ]
        rules += [
            (r"\b(%s)\(" % f, 1, STYLES["function"])
            for f in PythonHighlighter.functions
        ]

        # All other rules
        rules += [
            # 'self'
            (r"\bself\b", 0, STYLES["self"]),
            # 'def' followed by an identifier
            (r"\bdef\b\s*(\w+)", 1, STYLES["defclass"]),
            # 'class' followed by an identifier
            (r"\bclass\b\s*(\w+)", 1, STYLES["defclass"]),
            # Numeric literals
            (r"\b[+-]?[0-9]+[lL]?\b", 0, STYLES["numbers"]),
            (r"\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b", 0, STYLES["numbers"]),
            (
                r"\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b",
                0,
                STYLES["numbers"],
            ),
            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES["string"]),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES["string"]),
            # From '#' until a newline
            (r"#[^\n]*", 0, STYLES["comment"]),
        ]

        # Build a QRegExp for each pattern
        self.rules = [
            (QtCore.QRegExp(pat), index, fmt) for (pat, index, fmt) in rules
        ]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        self.tripleQuoutesWithinStrings = []
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            index = expression.indexIn(text, 0)
            if index >= 0:
                # if there is a string we check
                # if there are some triple quotes within the string
                # they will be ignored if they are matched again
                if expression.pattern() in [
                    r'"[^"\\]*(\\.[^"\\]*)*"',
                    r"'[^'\\]*(\\.[^'\\]*)*'",
                ]:
                    innerIndex = self.tri_single[0].indexIn(text, index + 1)
                    if innerIndex == -1:
                        innerIndex = self.tri_double[0].indexIn(
                            text, index + 1
                        )

                    if innerIndex != -1:
                        tripleQuoteIndexes = range(innerIndex, innerIndex + 3)
                        self.tripleQuoutesWithinStrings.extend(
                            tripleQuoteIndexes
                        )

            while index >= 0:
                # skipping triple quotes within strings
                if index in self.tripleQuoutesWithinStrings:
                    index += 1
                    expression.indexIn(text, index)
                    continue

                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, format[self.gui.dark])
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegExp`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            start = delimiter.indexIn(text)
            # skipping triple quotes within strings
            if start in self.tripleQuoutesWithinStrings:
                return False
            # Move past this match
            add = delimiter.matchedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            end = delimiter.indexIn(text, start + add)
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + delimiter.matchedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = len(text) - start + add
            # Apply formatting
            self.setFormat(start, length, style[self.gui.dark])
            # Look for the next match
            start = delimiter.indexIn(text, start + length)

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False


class FortranNamelistHighlighter(QtGui.QSyntaxHighlighter):
    """Syntax highlighter for Fortran namelists."""

    # Fortran keywords
    keywords = [
        r"\.true\.",
        r"\.false\.",
    ]

    # Fortran operators
    operators = [
        "=",
        r"\%",
        r"\&",
        r"\/",
    ]

    # Fortran braces
    braces = [
        r"\(",
        r"\)",
        r"\[",
        r"\]",
    ]

    def __init__(self, gui, parent: QtGui.QTextDocument) -> None:
        super().__init__(parent)

        self.gui = gui

        rules = []

        # Keyword, operator and brace rules
        rules += [
            (r"%s" % w, 0, STYLES["keyword"])
            for w in FortranNamelistHighlighter.keywords
        ]
        rules += [
            (r"%s" % o, 0, STYLES["operator"])
            for o in FortranNamelistHighlighter.operators
        ]
        rules += [
            (r"%s" % b, 0, STYLES["brace"])
            for b in FortranNamelistHighlighter.braces
        ]

        # All other rules
        rules += [
            # '&' followed by an identifier
            (r"\&(\w+)", 1, STYLES["defclass"]),
            # Numeric literals
            (r"\b[+-]?[0-9]+[lL]?\b", 0, STYLES["numbers"]),
            (r"\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b", 0, STYLES["numbers"]),
            (
                r"\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b",
                0,
                STYLES["numbers"],
            ),
            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES["string"]),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES["string"]),
            # FIXME Hack: First part of double-quoted multi-line string,
            #  possibly containing escape sequences
            (r'"[^&\\]*(\\.[^&\\]*)*&', 0, STYLES["string"]),
            # FIXME Hack: First part of single-quoted multi-line string,
            #  possibly containing escape sequences
            (r"'[^&\\]*(\\.[^&\\]*)*&", 0, STYLES["string"]),
            # FIXME Hack: Middle part of multi-line string,
            #  possibly containing escape sequences
            (r"&[^&\\]*(\\.[^&\\]*)*&", 0, STYLES["string"]),
            # FIXME Hack: Last part of double-quoted multi-line string,
            #  possibly containing escape sequences
            (r'&[^"\\]*(\\.[^"\\]*)*"', 0, STYLES["string"]),
            # FIXME Hack: Last part of single-quoted multi-line string,
            #  possibly containing escape sequences
            (r"&[^'\\]*(\\.[^'\\]*)*'", 0, STYLES["string"]),
            # From '!' until a newline
            (r"![^\n]*", 0, STYLES["comment"]),
        ]

        # Build a QRegExp for each pattern
        self.rules = [
            (
                QtCore.QRegExp(pat, QtCore.Qt.CaseSensitivity.CaseInsensitive),
                index,
                fmt,
            )
            for (pat, index, fmt) in rules
        ]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        self.tripleQuoutesWithinStrings = []
        # Do other syntax formatting
        for expression, nth, format in self.rules:
            index = expression.indexIn(text, 0)

            while index >= 0:
                # We actually want the index of the nth match
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, format[self.gui.dark])
                index = expression.indexIn(text, index + length)

        self.setCurrentBlockState(0)
