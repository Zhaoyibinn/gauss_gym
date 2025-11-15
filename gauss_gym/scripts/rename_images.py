#!/usr/bin/env python3
"""Rename image files to zero-padded integers."""

import argparse
from pathlib import Path
from typing import List


def collect_files(target_dir: Path) -> List[Path]:
    """Return a sorted list of files in the directory."""
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {target_dir}")
    return sorted([p for p in target_dir.iterdir() if p.is_file()])


def rename_files(files: List[Path], padding: int) -> None:
    if not files:
        print("No files found; nothing to do.")
        return

    temp_files: List[Path] = []
    # First pass: move to temporary names to avoid collisions.
    for index, path in enumerate(files):
        temp_name = path.with_name(f"__tmp__{index:0{padding}d}{path.suffix}")
        counter = 0
        while temp_name.exists():
            counter += 1
            temp_name = path.with_name(
                f"__tmp__{index:0{padding}d}_{counter}{path.suffix}"
            )
        path.rename(temp_name)
        temp_files.append(temp_name)

    # Second pass: rename to final zero-padded names.
    for index, path in enumerate(temp_files):
        final_name = path.with_name(f"{index:0{padding}d}{path.suffix}")
        if final_name.exists():
            raise FileExistsError(f"Target file already exists: {final_name}")
        path.rename(final_name)

    print(f"Renamed {len(files)} files with padding {padding}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("dataset/zyb_home_own/images"),
        help="Directory containing the images to rename.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=4,
        help="Number of digits for zero-padding (default: 4).",
    )

    args = parser.parse_args()
    files = collect_files(args.directory)
    rename_files(files, padding=args.padding)


if __name__ == "__main__":
    main()
