"""Check or reverse the 2026-10-18 data consolidation; never delete datasets."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--check", action="store_true", help="Read-only validation")
    options.add_argument(
        "--restore", action="store_true", help="Undo recorded data moves"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/consolidation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("restored"):
        parser.exit(1, "The data moves have already been restored.\n")
    problems = []
    for move in manifest["moves"]:
        original = root / move["original"]
        canonical = root / move["canonical"]
        if not original.resolve().is_relative_to(
            root
        ) or not canonical.resolve().is_relative_to(root):
            problems.append(f"Path escapes workspace: {move['original']}")
        elif move.get("compatibility_alias_removed"):
            if original.exists() or original.is_symlink() or not canonical.exists():
                problems.append(
                    f"Missing destination or conflicting original path: {move['original']}"
                )
        elif (
            not original.is_symlink()
            or not canonical.exists()
            or original.resolve() != canonical.resolve()
        ):
            problems.append(f"Missing/changed compatibility alias: {move['original']}")
    if problems:
        parser.exit(1, "Refusing operation:\n" + "\n".join(problems) + "\n")
    if args.check:
        print(
            f"Verified {len(manifest['moves'])} data moves and their alias policies; no changes made."
        )
        return
    # Each unlink removes only a verified compatibility symlink. The data at
    # its canonical destination is then renamed back, never removed or copied.
    for move in reversed(manifest["moves"]):
        original = root / move["original"]
        canonical = root / move["canonical"]
        if not move.get("compatibility_alias_removed"):
            original.unlink()
        canonical.rename(original)
    manifest["restored"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Restored original data locations. Current documentation was not changed.")


if __name__ == "__main__":
    main()
