from PyQt5 import QtCore, QtWidgets


def init_compile_gui(gui):
    app = QtWidgets.QApplication.instance()

    def closeTerminal(gui):
        terminal = gui.terminal_compile

        if getattr(terminal, "process", None) is not None:
            terminal.process.shouldRestart = False
            terminal.process.kill()

    def restartTerminal(gui):
        terminal = gui.terminal_compile

        if getattr(terminal, "process", None) is not None:
            if not terminal.process.shouldRestart:
                return

        terminal.process = QtCore.QProcess(terminal)
        terminal.process.shouldRestart = True
        terminal.process.finished.connect(lambda: restartTerminal(gui))
        terminal.process.start("urxvt", ["-embed", str(int(terminal.winId())), "+sb"])

    app.lastWindowClosed.connect(lambda: closeTerminal(gui))
    restartTerminal(gui)
