from pathlib import Path


# Return the resolved absolute str(Path) for path relative to the package root
def resource_path(path):
    return str((Path(__file__).parent / path).resolve())


# open(file) for a file path relative to the package root
def open_resource(file, *args, **kwargs):
    return open(resource_path(file), *args, **kwargs)
