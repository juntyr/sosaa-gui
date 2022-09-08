from pathlib import Path

from PyQt5 import QtWidgets


def browsePath(title, save=False, directory=False, origin=None):
    pathSelector = QtWidgets.QFileDialog()

    pathSelector.setAcceptMode(
        pathSelector.AcceptMode.AcceptSave
        if save
        else pathSelector.AcceptMode.AcceptOpen
    )

    pathSelector.setOption(pathSelector.DontUseNativeDialog)

    if not save:
        pathSelector.setFileMode(pathSelector.FileMode.ExistingFile)

    if directory:
        pathSelector.setFileMode(pathSelector.FileMode.Directory)
        pathSelector.setOption(pathSelector.ShowDirsOnly)
        pathSelector.setOption(pathSelector.DontResolveSymlinks)

    pathSelector.setWindowTitle(title)

    pathSelector.setDirectory(origin or str(Path.cwd()))

    if pathSelector.exec() == 1:
        path = Path(pathSelector.selectedFiles()[0]).resolve()
    else:
        path = None

    return path
