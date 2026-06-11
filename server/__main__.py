"""Run the app: python -m server [optional/path/to/file.nc]"""

import sys

import uvicorn

from .app import app, manager


def main():
    if len(sys.argv) > 1:
        meta = manager.open(sys.argv[1])
        print(f"Opened {meta['path']} ({meta['size_bytes'] / 1e6:.1f} MB)")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
