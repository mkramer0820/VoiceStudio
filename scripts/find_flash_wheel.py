#!/usr/bin/env python3
"""Find a prebuilt FlashAttention 2 wheel for a Colab (or any Linux x86_64) runtime.

A FlashAttention wheel loads only against the exact Python, torch and CUDA it was built for. FlashAttention publishes wheels for CUDA 12
only; the community repo mjun0812/flash-attention-prebuild-wheels also builds CUDA 13 and newer torch. This script lists the matching
files so the URL can go into the notebook's FLASH_WHEEL setting.

    python scripts/find_flash_wheel.py                          # in a Colab cell or shell: reads Python, torch and CUDA from the runtime
    python scripts/find_flash_wheel.py --python cp313 --torch 2.11 --cuda cu130

Needs the network (one GitHub API request, no token). See docs/engines/qwen3-tts-colab-runtime.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request

REPO = "mjun0812/flash-attention-prebuild-wheels"
NAME_RE = re.compile(r"^flash_attn-(?P<version>[\d.]+(?:\.post\d+)?)\+cu(?P<cuda>\d+)torch(?P<torch>[\d.]+)"
                     r"(?P<abi>cxx11abi(?:TRUE|FALSE))?-(?P<py>cp\d+t?)-(?P<abi_tag>cp\d+t?)-(?P<platform>[a-z0-9_]+)\.whl$")


def parse_name(name: str):
    """The parts of a wheel file name, or None when it is not a FlashAttention wheel."""
    m = NAME_RE.match(name)
    return None if not m else {"version": m["version"], "cuda": "cu" + m["cuda"], "torch": m["torch"], "python": m["py"], "abi": m["abi_tag"], "platform": m["platform"]}


def runtime_tags() -> dict:
    """Python, torch and CUDA of the interpreter running this script (needs torch)."""
    import torch

    return {"python": f"cp{sys.version_info[0]}{sys.version_info[1]}", "torch": ".".join(torch.__version__.split("+")[0].split(".")[:2]),
            "cuda": "cu" + str(torch.version.cuda).replace(".", "")}


def matches(releases: list, python: str, torch_mm: str, cuda: str) -> list:
    """Linux x86_64 wheels that fit, newest release first, as {name, url, version, release}. Free-threaded (cp313t) builds never match cp313."""
    out = []
    for rel in releases:
        for asset in rel.get("assets", []):
            p = parse_name(asset["name"])
            if p and p["python"] == python and p["abi"] == python and p["torch"] == torch_mm and p["cuda"] == cuda and p["platform"] == "linux_x86_64":
                out.append({"name": asset["name"], "version": p["version"], "release": rel["tag_name"],
                            "url": asset.get("browser_download_url") or f"https://github.com/{REPO}/releases/download/{rel['tag_name']}/"
                                   + urllib.parse.quote(asset["name"])})
    return sorted(out, key=lambda w: ([int(x) for x in re.findall(r"\d+", w["version"])], [int(x) for x in re.findall(r"\d+", w["release"])]), reverse=True)  # numeric: v0.9.31 beats v0.9.4


def fetch_releases() -> list:
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases?per_page=100", headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", help="e.g. cp313 (default: this interpreter)")
    ap.add_argument("--torch", help="torch major.minor, e.g. 2.11 (default: the installed torch)")
    ap.add_argument("--cuda", help="e.g. cu130 (default: the installed torch's CUDA)")
    args = ap.parse_args(argv)
    tags = {"python": args.python, "torch": args.torch, "cuda": args.cuda}
    if not all(tags.values()):
        auto = runtime_tags()
        tags = {k: v or auto[k] for k, v in tags.items()}
    print("looking for: Python %(python)s, torch %(torch)s, %(cuda)s, linux x86_64" % tags)
    found = matches(fetch_releases(), tags["python"], tags["torch"], tags["cuda"])
    if not found:
        print("No matching wheel. Options: pick an older Colab runtime version (Runtime > Change runtime type), or USE_VENV='auto' in the notebook "
              "(an isolated Python 3.12 + torch 2.8 environment with the official wheel), or ATTN_FALLBACK=True (PyTorch attention, no wheel).")
        return 1
    for w in found[:5]:
        print(f"  {w['name']}   ({w['release']})\n    {w['url']}")
    print("Put the first URL into FLASH_WHEEL = '...' in the notebook's parameters cell. FlashAttention 2 supports Ampere, Ada and Hopper GPUs only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
