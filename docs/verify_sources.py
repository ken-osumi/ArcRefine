"""Verify the included Mosaic source/weight hashes; run from any directory."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/vendor-provenance.json").read_text())
for name, expected in manifest["files"].items():
    with (root / name).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"Hash mismatch: {name}")
print(f"Verified {len(manifest['files'])} vendored source/weight files.")
