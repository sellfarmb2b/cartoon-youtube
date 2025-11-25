import os
import sys
import platform
from typing import Optional


def _base_path() -> str:
    """
    Returns the base path that should be used for resolving resources.
    When running inside a PyInstaller bundle, sys._MEIPASS points to the
    temporary extraction directory. Otherwise, we use the project root,
    which is assumed to be one level above this utils module.
    """
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resource_path(relative_path: str) -> str:
    """
    Resolve a relative path to an absolute path that works both in
    development and when bundled with PyInstaller.
    """
    base = _base_path()
    return os.path.normpath(os.path.join(base, relative_path))


def _ensure_executable(path: str) -> None:
    """Ensure the binary has executable permissions on Unix-like systems."""
    if platform.system().lower() == "windows":
        return
    try:
        current_mode = os.stat(path).st_mode
        os.chmod(path, current_mode | 0o111)
    except OSError:
        # If chmod fails we silently ignore—this typically means the file
        # already has the correct permissions or the filesystem is read-only.
        pass


def get_ffmpeg_path() -> str:
    """
    Return the absolute path to the FFmpeg binary for the current platform.
    macOS / Linux -> bin/mac/ffmpeg (or bin/bin:mac/ffmpeg if the folder name
    contains a colon on disk)
    Windows -> bin/win/ffmpeg.exe (or bin/bin:win/ffmpeg.exe)
    """
    system = platform.system().lower()
    if system == "windows":
        candidates = [
            os.path.join("bin", "win", "ffmpeg.exe"),
            os.path.join("bin", "bin:win", "ffmpeg.exe"),
        ]
    else:
        candidates = [
            os.path.join("bin", "mac", "ffmpeg"),
            os.path.join("bin", "bin:mac", "ffmpeg"),
        ]

    for relative_path in candidates:
        ffmpeg_path = resource_path(relative_path)
        if os.path.exists(ffmpeg_path):
            _ensure_executable(ffmpeg_path)
            return ffmpeg_path

    raise FileNotFoundError(
        "FFmpeg 실행 파일을 찾을 수 없습니다. "
        "bin/mac 또는 bin/win 폴더에 ffmpeg 바이너리가 있는지 확인하세요."
    )


def get_ffprobe_path(ffmpeg_path: Optional[str] = None) -> Optional[str]:
    """
    Attempt to locate an ffprobe binary next to ffmpeg. Some distributions
    bundle ffprobe separately; if it is not available we return None.
    """
    if ffmpeg_path is None:
        try:
            ffmpeg_path = get_ffmpeg_path()
        except FileNotFoundError:
            return None

    directory = os.path.dirname(ffmpeg_path)
    if platform.system().lower() == "windows":
        candidate = os.path.join(directory, "ffprobe.exe")
    else:
        candidate = os.path.join(directory, "ffprobe")

    if os.path.exists(candidate):
        _ensure_executable(candidate)
        return candidate

    return None

