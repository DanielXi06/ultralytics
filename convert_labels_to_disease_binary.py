from __future__ import annotations

import argparse
from pathlib import Path

# -----------------------------
# Direct-run configuration
# Fill these paths if you want to run this script directly from an IDE or by double-clicking it.
# Leave them empty if you prefer command-line arguments.
# -----------------------------
SOURCE_LABEL_DIR = r""
TARGET_LABEL_DIR = r""
ALLOW_OVERWRITE = True
PAUSE_AT_END = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert YOLO detection labels from multi-class disease IDs to a single "
            "binary disease class while preserving bounding boxes."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source label root directory. All .txt files under this directory will be processed recursively.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Destination label root directory. Converted labels will be written here with the same subdirectory layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing destination directory.",
    )
    return parser.parse_args()


def pause_if_needed() -> None:
    if PAUSE_AT_END:
        try:
            input("Press Enter to exit...")
        except EOFError:
            pass


def get_direct_run_args() -> argparse.Namespace | None:
    src = SOURCE_LABEL_DIR.strip()
    dst = TARGET_LABEL_DIR.strip()
    if not src or not dst:
        return None
    return argparse.Namespace(src=Path(src), dst=Path(dst), overwrite=ALLOW_OVERWRITE)


def convert_label_line(line: str, file_path: Path, line_number: int) -> str:
    stripped = line.strip()
    if not stripped:
        return ""

    parts = stripped.split()
    if len(parts) != 5:
        raise ValueError(
            f"{file_path} line {line_number}: expected 5 columns in YOLO detection format, got {len(parts)}."
        )

    # Collapse all disease subclasses into the single positive class "0".
    parts[0] = "0"
    return " ".join(parts)


def convert_one_file(src_file: Path, dst_file: Path) -> tuple[int, int]:
    text = src_file.read_text(encoding="utf-8").splitlines()
    converted_lines: list[str] = []
    object_count = 0

    for line_number, line in enumerate(text, start=1):
        converted = convert_label_line(line, src_file, line_number)
        if converted:
            converted_lines.append(converted)
            object_count += 1

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(converted_lines)
    if converted_lines:
        content += "\n"
    dst_file.write_text(content, encoding="utf-8")
    return len(text), object_count


def main() -> None:
    args = get_direct_run_args() or parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source label directory does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src}")
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"Destination directory already exists: {dst}. Use --overwrite to allow this.")

    txt_files = sorted(src.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt label files were found under: {src}")

    converted_file_count = 0
    converted_line_count = 0
    converted_object_count = 0

    for src_file in txt_files:
        relative_path = src_file.relative_to(src)
        dst_file = dst / relative_path
        line_count, object_count = convert_one_file(src_file, dst_file)
        converted_file_count += 1
        converted_line_count += line_count
        converted_object_count += object_count

    print(f"Converted {converted_file_count} label files.")
    print(f"Processed {converted_line_count} total label lines.")
    print(f"Kept {converted_object_count} bounding boxes and mapped all classes to 0.")
    print(f"Converted labels saved to: {dst}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pause_if_needed()
