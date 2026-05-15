#!/usr/bin/env python3
"""Normalize the Free3D active wheelchair download for local Isaac usage.

The Free3D archive is not committed because the source model is listed as
personal-use and this fork is public. This script expects the downloaded RAR in
the local asset folder, extracts it if possible, and writes normalized
ASCII/UTF-8 OBJ, MTL, and texture files under ``visual/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ARCHIVE_NAME = "y2ztdkf6sg00-ActiveWheelchair_likeKueschall.rar"
SOURCE_DIR_NAME = "Active Wheelchair_Küschall"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_asset_root() -> Path:
    return _repo_root() / "assets" / "objects" / "wheelchair" / "free3d_active_wheelchair"


def _extract_with_libarchive(archive: Path, extracted_root: Path) -> None:
    try:
        import libarchive
    except ImportError as exc:
        raise RuntimeError(
            "RAR extraction needs libarchive-c when bsdtar/unrar are unavailable. "
            "Create a temporary venv and install it with: python -m venv /tmp/libarchive-venv && "
            "/tmp/libarchive-venv/bin/pip install libarchive-c"
        ) from exc

    extracted_root.mkdir(parents=True, exist_ok=True)
    with libarchive.file_reader(str(archive)) as entries:
        for entry in entries:
            out_path = extracted_root / entry.pathname
            if entry.isdir:
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as out_file:
                for block in entry.get_blocks():
                    out_file.write(block)


def _find_source_dir(extracted_root: Path) -> Path:
    expected = extracted_root / SOURCE_DIR_NAME
    if expected.exists():
        return expected
    matches = [path for path in extracted_root.iterdir() if path.is_dir() and "Wheelchair" in path.name]
    if not matches:
        raise FileNotFoundError(f"Could not find extracted Free3D wheelchair folder under {extracted_root}")
    return matches[0]


def _copy_textures(source_dir: Path, visual_dir: Path) -> None:
    maps_dir = source_dir / "Küschall maps"
    copies = {
        maps_dir / "cloth2048.png": visual_dir / "camo_cloth_black_2048.png",
        maps_dir / "textur.bmp": visual_dir / "textur.bmp",
        maps_dir / "CHROMIC.JPG": visual_dir / "CHROMIC.JPG",
        maps_dir / "METAL7.JPG": visual_dir / "METAL7.JPG",
        maps_dir / "STEELPLT.JPG": visual_dir / "STEELPLT.JPG",
        maps_dir / "qualm.jpg": visual_dir / "qualm.jpg",
        maps_dir / "Lakerem.jpg": visual_dir / "Lakerem.jpg",
        source_dir / "Active Wheelchair.jpg": visual_dir / "active_wheelchair_preview.jpg",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src, dst)


def _normalize_mtl(source_dir: Path, visual_dir: Path) -> None:
    src = source_dir / "Küschall_obj" / "Rollstuhl_Küschall.mtl"
    dst = visual_dir / "active_wheelchair.mtl"
    text = src.read_text(encoding="latin-1", errors="replace")
    text = text.replace("cloth2048.png", "camo_cloth_black_2048.png")
    dst.write_text(text, encoding="utf-8")


def _normalize_obj(source_dir: Path, visual_dir: Path) -> tuple[list[float], list[float]]:
    src = source_dir / "Küschall_obj" / "Rollstuhl_Küschall.obj"
    dst = visual_dir / "active_wheelchair.obj"

    vertices: list[tuple[float, float, float]] = []
    with src.open("r", encoding="latin-1", errors="ignore") as in_file:
        for line in in_file:
            if line.startswith("v "):
                ox, oy, oz = (float(value) for value in line.split()[1:4])
                vertices.append((ox, oy, oz))

    if not vertices:
        raise ValueError(f"No vertices found in {src}")

    min_x = min(vertex[0] for vertex in vertices)
    max_x = max(vertex[0] for vertex in vertices)
    min_y = min(vertex[1] for vertex in vertices)
    max_y = max(vertex[1] for vertex in vertices)
    min_z = min(vertex[2] for vertex in vertices)
    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)
    scale = 0.0254  # source dimensions match inches; convert to meters.

    bounds_min = [float("inf"), float("inf"), float("inf")]
    bounds_max = [float("-inf"), float("-inf"), float("-inf")]
    with src.open("r", encoding="latin-1", errors="ignore") as in_file, dst.open("w", encoding="utf-8") as out_file:
        out_file.write("# Normalized active manual wheelchair visual mesh.\n")
        out_file.write("# Source: Free3D active-wheelchair-82422.\n")
        out_file.write("# Transform: inches to meters, local +X forward, +Y left, +Z up.\n")
        out_file.write("mtllib active_wheelchair.mtl\n")
        for line in in_file:
            if line.startswith("mtllib "):
                continue
            if line.startswith("v "):
                ox, oy, oz = (float(value) for value in line.split()[1:4])
                x = -(oy - mid_y) * scale
                y = (ox - mid_x) * scale
                z = (oz - min_z) * scale
                for index, value in enumerate((x, y, z)):
                    bounds_min[index] = min(bounds_min[index], value)
                    bounds_max[index] = max(bounds_max[index], value)
                out_file.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
            else:
                out_file.write(line)
    return bounds_min, bounds_max


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=_default_asset_root())
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--skip-extract", action="store_true", help="Use the existing extracted/ directory.")
    args = parser.parse_args()

    asset_root = args.asset_root.resolve()
    archive = args.archive or asset_root / "source" / ARCHIVE_NAME
    extracted_root = asset_root / "extracted"
    visual_dir = asset_root / "visual"

    if not args.skip_extract:
        if not archive.exists():
            raise FileNotFoundError(f"Missing Free3D archive: {archive}")
        _extract_with_libarchive(archive, extracted_root)

    source_dir = _find_source_dir(extracted_root)
    visual_dir.mkdir(parents=True, exist_ok=True)
    _copy_textures(source_dir, visual_dir)
    _normalize_mtl(source_dir, visual_dir)
    bounds_min, bounds_max = _normalize_obj(source_dir, visual_dir)
    size = [bounds_max[index] - bounds_min[index] for index in range(3)]

    print(f"Wrote normalized visual files to: {visual_dir}")
    print(f"Bounds min: {bounds_min}")
    print(f"Bounds max: {bounds_max}")
    print(f"Size meters: {size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
