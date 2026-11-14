from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

from .capture import PhotoCapture

_FO_DELETE = 3
_FOF_SILENT = 0x0004
_FOF_NOCONFIRMATION = 0x0010
_FOF_ALLOWUNDO = 0x0040
_FOF_NOERRORUI = 0x0400
_FOF_WANTNUKEWARNING = 0x4000


class _ShFileOperation(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", ctypes.c_ushort),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", wintypes.LPVOID),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


def _send_to_recycle_bin(path: str) -> bool:
    operation = _ShFileOperation(
        hwnd=None,
        wFunc=_FO_DELETE,
        pFrom=path + "\0\0",
        pTo=None,
        fFlags=(
            _FOF_SILENT
            | _FOF_NOCONFIRMATION
            | _FOF_ALLOWUNDO
            | _FOF_NOERRORUI
            | _FOF_WANTNUKEWARNING
        ),
    )
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    return result == 0 and not operation.fAnyOperationsAborted


def recycle_captured_photo(capture: PhotoCapture, *, send=None) -> bool:
    try:
        status = capture.path.stat()
    except OSError:
        return False
    if (status.st_dev, status.st_ino) != capture.file_id or not capture.path.is_file():
        return False
    if send is None:
        if os.name != "nt":
            return False
        send = _send_to_recycle_bin
    return bool(send(str(capture.path)))
