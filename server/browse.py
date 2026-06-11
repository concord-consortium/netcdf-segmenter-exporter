"""List directories server-side so the frontend can offer a file picker.

Browsers never reveal a picked file's real path to web pages, so the
server (which runs on the user's machine) does the browsing instead.
"""

from pathlib import Path

NETCDF_SUFFIXES = {".nc", ".nc4"}


def list_directory(path=None):
    """Return {path, parent, dirs, files} for a directory.

    path=None lists the user's home directory. Dotfiles are hidden, files
    are filtered to netCDF suffixes, and entries whose metadata can't be
    read (broken symlinks, unreadable mounts) are skipped.

    Raises FileNotFoundError, NotADirectoryError, or PermissionError
    (e.g. macOS privacy-protected folders).
    """
    directory = Path(path).expanduser() if path else Path.home()
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    dirs = []
    files = []
    for entry in directory.iterdir():  # raises PermissionError when blocked
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                dirs.append({"name": entry.name})
            elif entry.suffix.lower() in NETCDF_SUFFIXES and entry.is_file():
                files.append(
                    {"name": entry.name, "size_bytes": entry.stat().st_size}
                )
        except OSError:
            continue  # broken symlink or unreadable entry: skip it

    dirs.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: f["name"].lower())
    parent = None if directory.parent == directory else str(directory.parent)
    return {"path": str(directory), "parent": parent, "dirs": dirs, "files": files}
