"""
Encrypt small secrets (the AI key) for this Windows user, with Windows DPAPI.

    protect("sk-ant-...")   -> "dpapi:AQAAANCMnd8B..."   (safe to keep in QSettings)
    unprotect("dpapi:...")  -> "sk-ant-..."
    unprotect("sk-ant-...") -> "sk-ant-..."             (an old plain-text value passes through)

DPAPI (CryptProtectData) ties the encryption to the Windows login: another user
on the same PC, or a copy of the registry on another PC, can't decrypt it. No key
to manage and no new package: it is part of Windows, reached through ctypes.
Rejected: the `keyring` package (one more dependency for the same Windows
Credential Manager underneath), and our own AES key (it would have to be stored
somewhere too, which moves the problem instead of solving it).
https://learn.microsoft.com/windows/win32/api/dpapi/nf-dpapi-cryptprotectdata
"""
from __future__ import annotations

import base64
import ctypes
import sys
from ctypes import wintypes

PREFIX = "dpapi:"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _call(fn, data: bytes) -> bytes:
    src = _Blob(len(data), ctypes.cast(ctypes.create_string_buffer(data, len(data)), ctypes.POINTER(ctypes.c_char)))
    out = _Blob()
    # flags 1 = CRYPTPROTECT_UI_FORBIDDEN: never pop up a dialog
    if not fn(ctypes.byref(src), None, None, None, None, 1, ctypes.byref(out)):
        raise OSError(f"DPAPI failed (Windows error {ctypes.GetLastError()})")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)


def protect(text: str) -> str:
    """Encrypt for this Windows user. Empty stays empty."""
    if not text:
        return ""
    if sys.platform != "win32":
        return text   # ponytail: plain text off Windows; use the OS keychain there if the app is ever ported
    crypt = ctypes.windll.crypt32.CryptProtectData
    return PREFIX + base64.b64encode(_call(crypt, text.encode("utf-8"))).decode("ascii")


def unprotect(stored: str) -> str:
    """Decrypt a value from protect(); a value without the prefix (an old plain-text key) is returned as is.
    A value encrypted by another Windows user can't be read: returns "" (the user pastes the key again)."""
    if not stored.startswith(PREFIX):
        return stored
    try:
        raw = base64.b64decode(stored[len(PREFIX):])
        return _call(ctypes.windll.crypt32.CryptUnprotectData, raw).decode("utf-8")
    except (OSError, ValueError, AttributeError):
        return ""
