"""Phase 0: extract QUAM-AFM Lite (minus pickles / scripts) and summarize its layout.

The archive ships Python pickle dictionaries and a GUI script. Unpickling untrusted
data can execute code, so .pkl and .py members are skipped and never loaded.

Usage: python3 -I scripts/inspect_quam_lite.py <archive.tar.gz> <extract_dir>
"""
import sys
import tarfile
from collections import Counter
from pathlib import Path

SKIP_SUFFIXES = {".pkl", ".py", ".pyc"}


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with tarfile.open(archive) as tf:
        def members():
            nonlocal kept, skipped
            for m in tf:
                if Path(m.name).suffix.lower() in SKIP_SUFFIXES:
                    skipped += 1
                    print(f"  skipped {m.name}")
                    continue
                kept += 1
                yield m
        tf.extractall(dest, members=members(), filter="data")
    print(f"extracted {kept} members, skipped {skipped}")


def summarize(dest: Path) -> None:
    files = [p for p in dest.rglob("*") if p.is_file()]
    print(f"\n{len(files)} files")
    print("suffixes:", Counter(p.suffix.lower() for p in files).most_common())
    top = sorted({p.relative_to(dest).parts[:3] for p in files})
    dirs = Counter(parts[:2] for parts in top)
    print("\ntop-level layout (first two path levels, file-bearing subdir counts):")
    for k, v in sorted(dirs.items())[:40]:
        print(f"  {'/'.join(k)}: {v}")
    quam = next(dest.rglob("QUAM"), None)
    if quam is None:
        return
    kdirs = sorted(d for d in quam.iterdir() if d.is_dir() and d.name.startswith("K-"))
    print(f"\n{len(kdirs)} K-* parameter folders")
    if kdirs:
        mols = sorted(d for d in kdirs[0].iterdir() if d.is_dir())
        print(f"{kdirs[0].name}: {len(mols)} molecule folders; first: {mols[0].name}")
        for f in sorted(mols[0].iterdir()):
            print(f"  {f.name}  {f.stat().st_size} B")
    for extra in ("HM", "XYZ_FILES", "JMOL_IMAGES"):
        d = quam / extra
        if d.is_dir():
            ex = sorted(d.iterdir())
            print(f"{extra}: {len(ex)} entries; e.g. {[e.name for e in ex[:3]]}")


if __name__ == "__main__":
    archive, dest = Path(sys.argv[1]), Path(sys.argv[2])
    extract(archive, dest)
    summarize(dest)
