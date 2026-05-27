import json
import struct
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_master.contrib.broker.addinflatjy_broker import inspect_addinflatjy_dll

CLI_PATH = ROOT / "scripts" / "test_addinflatjy_probe.py"
spec = importlib.util.spec_from_file_location("test_addinflatjy_probe_cli", CLI_PATH)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def _build_minimal_pe() -> bytes:
    data = bytearray(0x1000)
    data[0:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, 0x14C, 2, 0, 0, 0, 0xE0, 0x210E)

    optional = 0x98
    struct.pack_into("<H", data, optional, 0x10B)
    struct.pack_into("<II", data, optional + 96, 0x1000, 40)
    struct.pack_into("<II", data, optional + 104, 0x2000, 40)

    section = 0x178
    data[section : section + 8] = b".rdata\0\0"
    struct.pack_into("<IIIIIIHHI", data, section + 8, 0x1000, 0x1000, 0x800, 0x200, 0, 0, 0, 0, 0)
    data[section + 40 : section + 48] = b".idata\0\0"
    struct.pack_into("<IIIIIIHHI", data, section + 48, 0x1000, 0x2000, 0x800, 0xA00, 0, 0, 0, 0, 0)

    export_dir = 0x200
    struct.pack_into("<IIHHIIIIIII", data, export_dir, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0x1040, 0)
    struct.pack_into("<II", data, 0x240, 0x1060, 0x1070)
    data[0x260 : 0x260 + len(b"LOAD\0")] = b"LOAD\0"
    data[0x270 : 0x270 + len(b"DoBuyOrder\0")] = b"DoBuyOrder\0"

    import_dir = 0xA00
    struct.pack_into("<IIIII", data, import_dir, 0, 0, 0, 0x2028, 0)
    data[0xA28 : 0xA28 + len(b"KERNEL32.dll\0")] = b"KERNEL32.dll\0"
    return bytes(data)


def test_inspect_addinflatjy_parses_exports_and_dependencies(tmp_path):
    dll_path = tmp_path / "AddinFlatJy.dll"
    dll_path.write_bytes(_build_minimal_pe())

    payload = inspect_addinflatjy_dll(str(dll_path))

    assert payload["exists"] is True
    assert payload["pe_bits"] == 32
    assert payload["pe_machine"] == "x86"
    assert payload["exports"] == ["DoBuyOrder", "LOAD"]
    assert payload["dependencies"] == ["KERNEL32.dll"]
    assert payload["safe_candidates"] == ["LOAD"]
    assert payload["trading_candidates"] == ["DoBuyOrder"]
    assert payload["loaded"] is False


def test_inspect_addinflatjy_missing_dll_is_structured(tmp_path):
    payload = inspect_addinflatjy_dll(str(tmp_path / "missing.dll"))

    assert payload["exists"] is False
    assert "DLL not found" in payload["error"]
    assert payload["exports"] == []


def test_cli_outputs_json_for_functions_probe(monkeypatch, capsys, tmp_path):
    dll_path = tmp_path / "AddinFlatJy.dll"
    dll_path.write_bytes(_build_minimal_pe())
    monkeypatch.setattr("sys.argv", ["test_addinflatjy_probe.py", "--dll-path", str(dll_path)])

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["probe"] == "functions"
    assert payload["export_count"] == 2
    assert payload["dependency_count"] == 1


def test_c_helper_object_probe_is_read_only():
    source = (ROOT / "scripts" / "addinflatjy_probe.c").read_text(encoding="utf-8")

    assert "--object" in source
    assert "Addin_GetObject" in source
    assert "get_object();" in source
    assert "no_vtable_calls=1" in source
    assert "no_feature_calls=1" in source
    assert "no_buy_sell_calls=1" in source
    assert "VTABLE_FN" in source
    assert "vtable_words[i](" not in source
    for forbidden_export in ('"BUY"', '"SELL"', '"Buy"', '"Sell"', '"Feature"', '"Live"', '"live"'):
        assert forbidden_export not in source
