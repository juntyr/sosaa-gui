from typing import Any, Callable

from PyQt5.QtCore import QObject, QThread, pyqtSignal


def run_in_thread(
    task: Callable[[], Any],
    on_success: Callable[[Any], None],
    on_exception: Callable[[Exception], None],
) -> QThread:
    thread = QThread()
    thread.my_worker = _Worker(task)
    thread.my_worker.moveToThread(thread)

    thread.started.connect(thread.my_worker.run)
    thread.my_worker.on_finished.connect(thread.quit)
    thread.my_worker.on_finished.connect(thread.my_worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.my_worker.on_success.connect(on_success)
    thread.my_worker.on_exception.connect(on_exception)

    thread.start()

    return thread


class _Worker(QObject):
    on_success = pyqtSignal(object)
    on_exception = pyqtSignal(Exception)
    on_finished = pyqtSignal()

    def __init__(self, task: Callable[[], Any], *args, **kwargs):
        self.task = task

        super().__init__(*args, **kwargs)

    def run(self):
        try:
            result = self.task()
            self.on_success.emit(result)
        except Exception as err:
            self.on_exception.emit(err)
        finally:
            self.on_finished.emit()
