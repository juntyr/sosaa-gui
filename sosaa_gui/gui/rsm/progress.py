from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class _RsmProgressInner(QObject):
    def __init__(self, progress):
        super().__init__()

        self.progress = progress

    @pyqtSlot(object, object, object, object)
    def update(self, value=None, min=None, max=None, format=None):
        if min is not None:
            self.progress.setMinimum(min)

        if max is not None:
            self.progress.setMaximum(max)

        if format is not None:
            self.progress.setFormat(format)

        if value is not None:
            self.progress.setValue(value)
        else:
            self.progress.setValue(self.progress.value() + 1)


class RsmProgress(QObject):
    on_update = pyqtSignal(object, object, object, object)

    def __init__(self, progress):
        super().__init__()

        self.inner = _RsmProgressInner(progress)

        self.on_update.connect(self.inner.update)

    def update(self, value=None, min=None, max=None, format=None):
        self.on_update.emit(value, min, max, format)


class RsmMajorMinorProgress:
    def __init__(self, major, minor):
        self.minor = RsmProgress(minor)
        self.major = RsmProgress(major)

    def update_minor(self, *args, **kwargs):
        self.minor.update(*args, **kwargs)

    def update_major(self, *args, **kwargs):
        self.major.update(*args, **kwargs)
