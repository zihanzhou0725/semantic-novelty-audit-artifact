"""Restore optional test inputs using only the Python standard library."""

import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
    for line in (ROOT / "OPTIONAL_TEST_INPUTS_SHA256SUMS").read_text().splitlines()
    if line.strip()
}

with tarfile.open(ROOT / "OPTIONAL_TEST_INPUTS.tar.xz", mode="r:xz") as archive:
    members = archive.getmembers()
    if {m.name for m in members} != set(EXPECTED):
        raise ValueError("Archive members do not match the test-input manifest")
    for member in members:
        target = (ROOT / member.name).resolve()
        if not member.isfile() or not target.is_relative_to(ROOT):
            raise ValueError(f"Invalid archive member: {member.name}")
        content = archive.extractfile(member).read()
        if hashlib.sha256(content).hexdigest() != EXPECTED[member.name]:
            raise ValueError(f"Hash mismatch: {member.name}")
        if target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != EXPECTED[member.name]:
                raise ValueError(f"Refusing to replace a different file: {member.name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        print(f"Verified {member.name}")

print(f"Ready: {len(EXPECTED)} frozen test inputs restored.")
