from sosaa_gui.version import sosaa_version

__version__ = sosaa_version


def run():
    import sosaa_gui.__main__

    # Fake-use the main import
    sosaa_gui.__main__
