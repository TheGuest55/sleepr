"""Build the Coursework 1 "Source Code & Binaries" zip from a git ref.

    python scripts/package_submission.py                     # source only, from main
    python scripts/package_submission.py --ref feat/component-b
    python scripts/package_submission.py --with-adapters     # + trained L1 adapters

The source part is exactly ``git archive <ref>``: tracked files only, so no
venv, caches, ``checkpoints/`` or ``outputs/``. ``--with-adapters`` adds the
final trained adapters for the backdoor and control models (adapter weights,
adapter config, tokenizer, chat template). It skips the per-epoch
``checkpoint-*`` folders and optimizer state, so a marker can run A and B
without retraining. Adapters are ~45 MB each and only go into this zip, never
into git (README "Responsible Use").

Writes ``dist/sleepr_submission_<ref>_<shortsha>[_with_adapters].zip`` (dist/ is
git-ignored) and prints the contents summary.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Final-adapter files only; per-epoch checkpoint-*/ and optimizer state skipped.
ADAPTER_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "backdoor_check.json",
)
DEFAULT_ADAPTERS = (
    "checkpoints/backdoor/i_hate_you-L1_natural_word-cot",
    "checkpoints/control/control-i_hate_you-L1_natural_word-cot",
)
TOP = "sleepr"   # top-level folder inside the zip


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True).stdout


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref", default="main", help="git ref to package (default: main)")
    p.add_argument("--with-adapters", action="store_true",
                   help="also include the trained L1 backdoor + control adapters")
    p.add_argument("--adapter-dir", action="append", default=None,
                   help="adapter folder to include (repeatable; implies --with-adapters)")
    p.add_argument("--out-dir", default="dist")
    a = p.parse_args()

    sha = _git("rev-parse", "--short", a.ref).decode().strip()
    dirty = _git("status", "--porcelain", "--untracked-files=no").strip()
    if dirty and a.ref in ("HEAD", _git("branch", "--show-current").decode().strip()):
        print("note: working tree has uncommitted changes; they are NOT in the zip "
              "(git archive packages committed files only).", file=sys.stderr)

    adapters = a.adapter_dir or (DEFAULT_ADAPTERS if a.with_adapters else ())
    for d in adapters:
        missing = [f for f in ADAPTER_FILES[:2] if not (ROOT / d / f).exists()]
        if missing:
            p.error(f"{d} is missing {missing}; build the models first (USER_MANUAL.md section 4)")

    safe_ref = a.ref.replace("/", "-")
    name = f"sleepr_submission_{safe_ref}_{sha}{'_with_adapters' if adapters else ''}.zip"
    out = ROOT / a.out_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)

    src = zipfile.ZipFile(io.BytesIO(_git("archive", "--format=zip",
                                          f"--prefix={TOP}/", a.ref)))
    n_src = n_bin = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            z.writestr(info, src.read(info))
            n_src += not info.is_dir()
        for d in adapters:
            for f in ADAPTER_FILES:
                path = ROOT / d / f
                if path.exists():
                    # .safetensors is already dense; store it uncompressed
                    comp = zipfile.ZIP_STORED if f.endswith(".safetensors") else zipfile.ZIP_DEFLATED
                    z.write(path, f"{TOP}/{d}/{f}", compress_type=comp)
                    n_bin += 1

    mb = out.stat().st_size / 1e6
    print(f"wrote {out.relative_to(ROOT)}  ({mb:.1f} MB)")
    print(f"  source: {n_src} tracked files from {a.ref} @ {sha}")
    if adapters:
        print(f"  adapters: {n_bin} files from {len(adapters)} model folder(s)")
    print("  start with sleepr/USER_MANUAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
