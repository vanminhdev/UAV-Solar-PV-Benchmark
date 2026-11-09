#!/usr/bin/env python3
"""
Build a portable single-file HTML version of label_tool.html with embedded
JPEG images (from <package_dir>/images/*.jpg), optional classes.txt, and
optional preset annotator code.

Usage:
    python build_portable.py <package_dir> <out_html> [--code A2]
"""

import argparse
import base64
import json
from pathlib import Path
import sys


def load_package_images(package_dir: Path) -> dict[str, str]:
    """Read ONLY images/*.jpg inside package_dir and return {stem: data_url}."""
    images_dir = package_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing 'images/' directory in {package_dir}")

    jpg_files = sorted(
        (
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".jpg"
        ),
        key=lambda p: p.name.lower(),
    )
    if not jpg_files:
        raise ValueError(f"No .jpg files found in {images_dir}")

    embedded: dict[str, str] = {}
    for img_path in jpg_files:
        stem = img_path.stem
        b64_data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        embedded[stem] = f"data:image/jpeg;base64,{b64_data}"
    return embedded


def load_package_classes(package_dir: Path) -> list[str] | None:
    """Read ONLY <package_dir>/classes.txt if present."""
    classes_path = package_dir / "classes.txt"
    if not classes_path.is_file():
        return None
    raw_text = classes_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return lines if lines else None


def build_portable_html(
    package_dir: Path, out_html: Path, annotator_code: str = ""
) -> int:
    script_dir = Path(__file__).resolve().parent
    template_path = script_dir / "label_tool.html"
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")

    html = template_path.read_text(encoding="utf-8")
    if "/*EMBEDDED_IMAGES*/" not in html:
        raise RuntimeError(
            "Placeholder /*EMBEDDED_IMAGES*/ not found in label_tool.html"
        )

    embedded_images = load_package_images(package_dir)
    embedded_classes = load_package_classes(package_dir)

    images_json = json.dumps(embedded_images, ensure_ascii=False)
    html = html.replace(
        "/*EMBEDDED_IMAGES*/ null",
        f"/*EMBEDDED_IMAGES*/ {images_json}",
        1,
    )

    if embedded_classes is not None and "/*EMBEDDED_CLASSES*/ null" in html:
        classes_json = json.dumps(embedded_classes, ensure_ascii=False)
        html = html.replace(
            "/*EMBEDDED_CLASSES*/ null",
            f"/*EMBEDDED_CLASSES*/ {classes_json}",
            1,
        )

    if "/*EMBEDDED_CODE*/ \"\"" in html:
        code_json = json.dumps(annotator_code or "", ensure_ascii=False)
        html = html.replace(
            "/*EMBEDDED_CODE*/ \"\"",
            f"/*EMBEDDED_CODE*/ {code_json}",
            1,
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_bytes = html.encode("utf-8")
    out_html.write_bytes(out_bytes)
    return len(out_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed package images and classes into a portable HTML file."
    )
    parser.add_argument(
        "package_dir",
        type=Path,
        help="Path to package directory containing images/*.jpg and optional classes.txt",
    )
    parser.add_argument(
        "out_html",
        type=Path,
        help="Output path for the generated portable HTML file",
    )
    parser.add_argument(
        "--code",
        default="",
        help="Preset annotator code (e.g. A2)",
    )
    args = parser.parse_args()

    size_bytes = build_portable_html(args.package_dir, args.out_html, args.code)
    print(f"Built {args.out_html} ({size_bytes} bytes, {size_bytes / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
