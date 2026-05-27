# Copyright (c) QuantMaster Contributors.
# Licensed under the MIT License.

"""Safe AddinFlatJy.dll probe utilities.

This module intentionally does not implement live trading.  It can inspect the
PE export/import tables and optionally ask Windows to load the DLL so missing
side-by-side dependencies can be diagnosed.
"""

from __future__ import annotations

import ctypes
import json
import platform
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_DLL_PATH = Path(r"C:\silkriver\TCPlugins\AddinFlatJy.dll")
SAFE_FUNCTION_HINTS = ("FUNCTIONS", "LOAD", "VERSION", "INIT")
TRADING_FUNCTION_HINTS = ("BUY", "SELL", "ORDER", "CANCEL", "GRIDJY", "LEVIN", "TRADE", "XIADAN")


class PEFormatError(ValueError):
    """Raised when a DLL cannot be parsed as a PE image."""


@dataclass(frozen=True)
class PESection:
    name: str
    virtual_address: int
    virtual_size: int
    raw_pointer: int
    raw_size: int


def inspect_addinflatjy_dll(dll_path: Optional[str] = None, *, load: bool = False) -> Dict[str, object]:
    """Return a JSON-friendly AddinFlatJy probe payload.

    ``load=True`` only calls ``ctypes.WinDLL``; no exported function is invoked.
    On a 64-bit Python process, loading a 32-bit trading DLL is expected to fail
    with a bad-image error, while export/dependency parsing still works.
    """

    path = Path(dll_path) if dll_path else DEFAULT_DLL_PATH
    payload: Dict[str, object] = {
        "dll_path": str(path),
        "exists": path.exists(),
        "python_bits": struct.calcsize("P") * 8,
        "machine": platform.machine(),
        "loaded": False,
        "load_error": None,
        "exports": [],
        "dependencies": [],
        "safe_candidates": [],
        "trading_candidates": [],
    }
    if not path.exists():
        payload["error"] = f"DLL not found: {path}"
        return payload

    try:
        pe = PEImage(path)
        exports = pe.exports()
        dependencies = pe.imported_dlls()
        payload.update(
            {
                "pe_machine": pe.machine_name,
                "pe_bits": pe.bits,
                "exports": exports,
                "export_count": len(exports),
                "dependencies": dependencies,
                "dependency_count": len(dependencies),
                "safe_candidates": _filter_names(exports, SAFE_FUNCTION_HINTS),
                "trading_candidates": _filter_names(exports, TRADING_FUNCTION_HINTS),
            }
        )
    except Exception as exc:
        payload["parse_error"] = f"{type(exc).__name__}: {exc}"

    if load:
        payload.update(load_addinflatjy_dll(path))
    return payload


def load_addinflatjy_dll(path: Path) -> Dict[str, object]:
    """Load the DLL without calling exports and return diagnostics."""

    if platform.system().lower() != "windows":
        return {"loaded": False, "load_error": "Load probe is only supported on Windows"}
    try:
        ctypes.WinDLL(str(path))
    except OSError as exc:
        return {"loaded": False, "load_error": f"{type(exc).__name__}: {exc}"}
    return {"loaded": True, "load_error": None}


def format_probe_result(payload: Dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _filter_names(names: Iterable[str], hints: Iterable[str]) -> List[str]:
    upper_hints = tuple(hint.upper() for hint in hints)
    return [name for name in names if any(hint in name.upper() for hint in upper_hints)]


class PEImage:
    """Small PE reader focused on exports and imported DLL names."""

    MACHINE_NAMES = {
        0x014C: "x86",
        0x8664: "x64",
        0x01C0: "ARM",
        0xAA64: "ARM64",
    }

    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        self.machine = 0
        self.bits = 0
        self.sections: List[PESection] = []
        self.data_directories: List[tuple[int, int]] = []
        self._parse_headers()

    @property
    def machine_name(self) -> str:
        return self.MACHINE_NAMES.get(self.machine, f"0x{self.machine:04x}")

    def exports(self) -> List[str]:
        rva, size = self._directory(0)
        if not rva or not size:
            return []
        offset = self._rva_to_offset(rva)
        directory = self._unpack_from("<IIHHIIIIIII", offset)
        number_of_names = directory[7]
        address_of_names = directory[9]
        names_offset = self._rva_to_offset(address_of_names)
        exports: List[str] = []
        for index in range(number_of_names):
            name_rva = self._unpack_from("<I", names_offset + index * 4)[0]
            exports.append(self._read_c_string(self._rva_to_offset(name_rva)))
        return sorted(exports)

    def imported_dlls(self) -> List[str]:
        rva, size = self._directory(1)
        if not rva or not size:
            return []
        offset = self._rva_to_offset(rva)
        dlls: List[str] = []
        descriptor_size = 20
        for index in range(max(1, size // descriptor_size + 1)):
            descriptor = self._unpack_from("<IIIII", offset + index * descriptor_size)
            if descriptor == (0, 0, 0, 0, 0):
                break
            name_rva = descriptor[3]
            if name_rva:
                dlls.append(self._read_c_string(self._rva_to_offset(name_rva)))
        return sorted(dict.fromkeys(dlls), key=str.lower)

    def _parse_headers(self) -> None:
        if self.data[:2] != b"MZ":
            raise PEFormatError("missing MZ header")
        pe_offset = self._unpack_from("<I", 0x3C)[0]
        if self.data[pe_offset : pe_offset + 4] != b"PE\0\0":
            raise PEFormatError("missing PE signature")

        coff_offset = pe_offset + 4
        self.machine, section_count, _timestamp, _symbols, _symbol_count, optional_size, _chars = self._unpack_from(
            "<HHIIIHH", coff_offset
        )
        optional_offset = coff_offset + 20
        magic = self._unpack_from("<H", optional_offset)[0]
        if magic == 0x10B:
            self.bits = 32
            data_directory_offset = optional_offset + 96
        elif magic == 0x20B:
            self.bits = 64
            data_directory_offset = optional_offset + 112
        else:
            raise PEFormatError(f"unsupported optional header magic: 0x{magic:04x}")

        directory_count = max(0, min(16, (optional_size - (data_directory_offset - optional_offset)) // 8))
        self.data_directories = [
            self._unpack_from("<II", data_directory_offset + index * 8) for index in range(directory_count)
        ]

        section_offset = optional_offset + optional_size
        for index in range(section_count):
            item = self._unpack_from("<8sIIIIIIHHI", section_offset + index * 40)
            name = item[0].split(b"\0", 1)[0].decode("ascii", errors="replace")
            self.sections.append(
                PESection(
                    name=name,
                    virtual_size=item[1],
                    virtual_address=item[2],
                    raw_size=item[3],
                    raw_pointer=item[4],
                )
            )

    def _directory(self, index: int) -> tuple[int, int]:
        if index >= len(self.data_directories):
            return 0, 0
        return self.data_directories[index]

    def _rva_to_offset(self, rva: int) -> int:
        for section in self.sections:
            size = max(section.virtual_size, section.raw_size)
            if section.virtual_address <= rva < section.virtual_address + size:
                return section.raw_pointer + (rva - section.virtual_address)
        if 0 <= rva < len(self.data):
            return rva
        raise PEFormatError(f"RVA out of range: 0x{rva:x}")

    def _read_c_string(self, offset: int) -> str:
        end = self.data.find(b"\0", offset)
        if end < 0:
            raise PEFormatError(f"unterminated string at offset 0x{offset:x}")
        return self.data[offset:end].decode("ascii", errors="replace")

    def _unpack_from(self, fmt: str, offset: int):
        size = struct.calcsize(fmt)
        if offset < 0 or offset + size > len(self.data):
            raise PEFormatError(f"read out of range at offset 0x{offset:x}")
        return struct.unpack_from(fmt, self.data, offset)
