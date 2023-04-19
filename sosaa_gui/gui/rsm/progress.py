class RsmProgress:
    def __init__(self, progress):
        self.progress = progress

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

class RsmMajorMinorProgress:
    def __init__(self, major, minor):
        self.minor = RsmProgress(minor)
        self.major = RsmProgress(major)

    def update_minor(self, *args, **kwargs):
        self.minor.update(*args, **kwargs)

    def update_major(self, *args, **kwargs):
        self.major.update(*args, **kwargs)
