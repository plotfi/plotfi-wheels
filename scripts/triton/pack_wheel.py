"""Pack already-built Triton artifacts into a wheel without re-running cmake.

Usage:
    python pack_wheel.py [-o OUTPUT_DIR]

Requires that triton has already been built (libtriton.so etc. exist).
"""

import argparse
import hashlib
import base64
import csv
import io
import os
import re
import shutil
import sys
import sysconfig
import zipfile
from pathlib import Path


def get_version():
    """Extract version from setup.py."""
    setup_py = Path(__file__).parent / "setup.py"
    content = setup_py.read_text()
    m = re.search(r'TRITON_VERSION\s*=\s*"([^"]+)"', content)
    base = m.group(1) if m else "0.0.0"
    # The version in setup.py includes the git suffix already via
    # get_triton_version_suffix(), but we're reading the string literal.
    # Add git hash ourselves.
    try:
        import subprocess
        h = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=str(Path(__file__).parent),
        ).decode().strip()
        return f"{base}+git{h}"
    except Exception:
        return base


def get_wheel_tag():
    vi = sys.version_info
    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"cp{vi.major}{vi.minor}-cp{vi.major}{vi.minor}-{plat}"


def sha256_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def collect_files(triton_root: Path):
    """Yield (arcname, src_path) pairs for all files to include."""
    python_dir = triton_root / "python"

    # Collect all triton Python packages
    triton_pkg = python_dir / "triton"
    for root, dirs, files in os.walk(triton_pkg):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            src = Path(root) / f
            # arcname relative to python_dir
            arcname = src.relative_to(python_dir)
            yield str(arcname), src

    # Backend dirs that are symlinks (nvidia, amd) — follow them
    for backend_name in ["nvidia", "amd"]:
        backend_dir = triton_pkg / "backends" / backend_name
        if backend_dir.exists():
            for root, dirs, files in os.walk(backend_dir, followlinks=True):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in files:
                    if f.endswith((".pyc", ".pyo")):
                        continue
                    src = Path(root) / f
                    arcname = src.relative_to(python_dir)
                    yield str(arcname), src

    # Language extra dirs (cuda, hip, tlx etc.)
    extra_dir = triton_pkg / "language" / "extra"
    if extra_dir.exists():
        for root, dirs, files in os.walk(extra_dir, followlinks=True):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                src = Path(root) / f
                arcname = src.relative_to(python_dir)
                yield str(arcname), src

    # Proton profiler
    proton_dir = triton_root / "third_party" / "proton" / "proton"
    if proton_dir.exists():
        for root, dirs, files in os.walk(proton_dir, followlinks=True):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                src = Path(root) / f
                rel = src.relative_to(proton_dir)
                arcname = Path("triton") / "profiler" / rel
                yield str(arcname), src


def build_wheel(output_dir: Path):
    triton_root = Path(__file__).parent.resolve()
    version = get_version()
    tag = get_wheel_tag()
    wheel_name = f"triton-{version}-{tag}.whl"

    # Strip local version (everything after +) from filename to avoid
    # issues with + in URLs (e.g. GitHub raw downloads).
    filename_version = version.split("+")[0]
    wheel_filename = f"triton-{filename_version}-{tag}.whl"
    dist_info = f"triton-{version}.dist-info"

    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = output_dir / wheel_filename

    print(f"Packing triton {version} -> {wheel_path}")

    # Collect all files, dedup by arcname
    seen = set()
    file_list = []
    for arcname, src in collect_files(triton_root):
        arcname_str = str(arcname)
        if arcname_str not in seen:
            seen.add(arcname_str)
            file_list.append((arcname_str, src))

    # Build the wheel
    record_entries = []

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in file_list:
            data = src.read_bytes()
            zf.writestr(arcname, data)
            digest = sha256_digest(data)
            record_entries.append((arcname, f"sha256={digest}", str(len(data))))

        # METADATA
        metadata = f"""\
Metadata-Version: 2.1
Name: triton
Version: {version}
Summary: A language and compiler for custom Deep Learning operations
Home-page: https://github.com/triton-lang/triton/
Author: Philippe Tillet
Author-email: phil@openai.com
License: MIT
""".encode()
        meta_name = f"{dist_info}/METADATA"
        zf.writestr(meta_name, metadata)
        record_entries.append((meta_name, f"sha256={sha256_digest(metadata)}", str(len(metadata))))

        # WHEEL
        wheel_meta = f"""\
Wheel-Version: 1.0
Generator: pack_wheel.py
Root-Is-Purelib: false
Tag: {tag}
""".encode()
        wheel_name_entry = f"{dist_info}/WHEEL"
        zf.writestr(wheel_name_entry, wheel_meta)
        record_entries.append((wheel_name_entry, f"sha256={sha256_digest(wheel_meta)}", str(len(wheel_meta))))

        # top_level.txt
        top_level = b"triton\n"
        tl_name = f"{dist_info}/top_level.txt"
        zf.writestr(tl_name, top_level)
        record_entries.append((tl_name, f"sha256={sha256_digest(top_level)}", str(len(top_level))))

        # entry_points.txt
        entry_points = b"""\
[console_scripts]
proton-viewer = triton.profiler.viewer:main
proton = triton.profiler.proton:main

[triton.backends]
nvidia = triton.backends.nvidia
amd = triton.backends.amd
"""
        ep_name = f"{dist_info}/entry_points.txt"
        zf.writestr(ep_name, entry_points)
        record_entries.append((ep_name, f"sha256={sha256_digest(entry_points)}", str(len(entry_points))))

        # RECORD (must be last, no hash for itself)
        buf = io.StringIO()
        writer = csv.writer(buf)
        for row in record_entries:
            writer.writerow(row)
        record_name = f"{dist_info}/RECORD"
        writer.writerow((record_name, "", ""))
        record_data = buf.getvalue().encode()
        zf.writestr(record_name, record_data)

    size_mb = wheel_path.stat().st_size / (1024 * 1024)
    print(f"Created {wheel_path} ({size_mb:.1f} MB)")
    print(f"  Files: {len(file_list)}")
    print(f"  Tag: {tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack pre-built Triton into a wheel")
    parser.add_argument("-o", "--output-dir", default="triton-wheel", help="Output directory")
    args = parser.parse_args()
    build_wheel(Path(args.output_dir))
