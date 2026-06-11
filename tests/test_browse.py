import os
from pathlib import Path

import pytest

from server.browse import list_directory


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "beta").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / "Zebra.NC").write_bytes(b"x" * 10)
    (tmp_path / "apple.nc").write_bytes(b"x" * 5)
    (tmp_path / "data.nc4").write_bytes(b"x" * 7)
    (tmp_path / "notes.txt").write_text("not netcdf")
    (tmp_path / ".hidden.nc").write_bytes(b"x")
    return tmp_path


def test_lists_dirs_and_netcdf_files_sorted(tree):
    out = list_directory(tree)
    assert out["path"] == str(tree.resolve())
    assert [d["name"] for d in out["dirs"]] == ["Alpha", "beta"]
    assert [f["name"] for f in out["files"]] == ["apple.nc", "data.nc4", "Zebra.NC"]
    assert out["files"][0]["size_bytes"] == 5


def test_excludes_non_netcdf_and_dotfiles(tree):
    out = list_directory(tree)
    names = [f["name"] for f in out["files"]] + [d["name"] for d in out["dirs"]]
    assert "notes.txt" not in names
    assert ".hidden.nc" not in names
    assert ".hidden_dir" not in names


def test_parent_of_subdir_and_null_at_root(tree):
    out = list_directory(tree / "Alpha")
    assert out["parent"] == str(tree.resolve())
    root = list_directory("/")
    assert root["parent"] is None


def test_default_is_home():
    out = list_directory()
    assert out["path"] == str(Path.home())


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        list_directory("/nope/missing")


def test_file_path_raises(tree):
    with pytest.raises(NotADirectoryError):
        list_directory(tree / "apple.nc")


def test_broken_symlink_skipped(tree):
    os.symlink(tree / "gone.nc", tree / "broken.nc")
    out = list_directory(tree)
    assert "broken.nc" not in [f["name"] for f in out["files"]]
