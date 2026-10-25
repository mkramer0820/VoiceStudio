"""scripts/find_flash_wheel.py: choosing a FlashAttention wheel by file name (no network)."""
from __future__ import annotations

import importlib.util
import os

SPEC = importlib.util.spec_from_file_location("find_flash_wheel", os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "find_flash_wheel.py"))
ffw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ffw)


def _rel(tag, *names):
    return {"tag_name": tag, "assets": [{"name": n} for n in names]}


RELEASES = [
    _rel("v0.9.4", "flash_attn-2.8.3+cu130torch2.11-cp313-cp313-linux_x86_64.whl", "flash_attn-2.8.3+cu130torch2.11-cp313-cp313t-linux_x86_64.whl",
         "flash_attn-2.8.3+cu128torch2.11-cp313-cp313-linux_x86_64.whl", "flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl"),
    _rel("v0.9.17", "flash_attn-2.8.3+cu130torch2.12-cp313-cp313-linux_x86_64.whl", "flash_attn-2.8.3+cu130torch2.11-cp313-cp313-win_amd64.whl"),
    _rel("v0.4.22", "flash_attn-2.8.1+cu130torch2.11-cp313-cp313-linux_x86_64.whl", "not_a_wheel.txt"),
]


def test_a_wheel_name_is_read_into_python_torch_and_cuda():
    p = ffw.parse_name("flash_attn-2.8.3+cu130torch2.11-cp313-cp313-linux_x86_64.whl")
    assert p == {"version": "2.8.3", "cuda": "cu130", "torch": "2.11", "python": "cp313", "abi": "cp313", "platform": "linux_x86_64"}
    assert ffw.parse_name("flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl")["torch"] == "2.8"
    assert ffw.parse_name("not_a_wheel.txt") is None


def test_only_the_exact_runtime_matches_and_the_newest_flash_version_comes_first():
    got = ffw.matches(RELEASES, "cp313", "2.11", "cu130")
    assert [w["name"] for w in got] == ["flash_attn-2.8.3+cu130torch2.11-cp313-cp313-linux_x86_64.whl",
                                        "flash_attn-2.8.1+cu130torch2.11-cp313-cp313-linux_x86_64.whl"]  # 2.8.3 before 2.8.1
    assert got[0]["url"].endswith("/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp313-cp313-linux_x86_64.whl")  # the + is encoded for the download
    assert ffw.matches(RELEASES, "cp313", "2.12", "cu130")[0]["release"] == "v0.9.17"  # a different torch needs a different wheel
    assert ffw.matches(RELEASES, "cp313", "2.11", "cu126") == [] and ffw.matches(RELEASES, "cp314", "2.11", "cu130") == []
    assert all("cp313t" not in w["name"] and "win_amd64" not in w["name"] for w in ffw.matches(RELEASES, "cp313", "2.11", "cu130"))
