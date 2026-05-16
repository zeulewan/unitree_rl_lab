#!/usr/bin/env python3
"""Render an asset-only turntable for the normalized Free3D wheelchair OBJ."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Material:
    kd: np.ndarray
    texture: Path | None = None


@dataclass(frozen=True)
class Triangle:
    vertex_ids: tuple[int, int, int]
    uv_ids: tuple[int | None, int | None, int | None]
    material: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_obj_path() -> Path:
    return (
        _repo_root()
        / "assets"
        / "objects"
        / "wheelchair"
        / "free3d_active_wheelchair"
        / "visual"
        / "active_wheelchair.obj"
    )


def _parse_mtl(mtl_path: Path) -> dict[str, Material]:
    materials: dict[str, Material] = {}
    current_name: str | None = None
    current_kd = np.array([0.65, 0.65, 0.65], dtype=np.float32)
    current_texture: Path | None = None

    def commit() -> None:
        nonlocal current_name, current_kd, current_texture
        if current_name is not None:
            materials[current_name] = Material(kd=current_kd.copy(), texture=current_texture)

    with mtl_path.open("r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "newmtl":
                commit()
                current_name = " ".join(parts[1:])
                current_kd = np.array([0.65, 0.65, 0.65], dtype=np.float32)
                current_texture = None
            elif parts[0] == "Kd" and len(parts) >= 4:
                current_kd = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
            elif parts[0] == "map_Kd" and len(parts) >= 2:
                current_texture = mtl_path.parent / parts[-1]
    commit()
    return materials


def _parse_face_token(token: str, vertex_count: int, uv_count: int) -> tuple[int, int | None]:
    pieces = token.split("/")
    vertex_id = int(pieces[0])
    if vertex_id < 0:
        vertex_id = vertex_count + vertex_id
    else:
        vertex_id -= 1

    uv_id: int | None = None
    if len(pieces) > 1 and pieces[1]:
        uv_id = int(pieces[1])
        if uv_id < 0:
            uv_id = uv_count + uv_id
        else:
            uv_id -= 1
    return vertex_id, uv_id


def _load_obj(obj_path: Path) -> tuple[np.ndarray, np.ndarray, list[Triangle], dict[str, Material]]:
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    triangles: list[Triangle] = []
    materials: dict[str, Material] = {}
    current_material = "default"

    with obj_path.open("r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "mtllib":
                materials.update(_parse_mtl(obj_path.parent / parts[-1]))
            elif parts[0] == "usemtl":
                current_material = " ".join(parts[1:])
            elif parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "vt" and len(parts) >= 3:
                uvs.append((float(parts[1]), float(parts[2])))
            elif parts[0] == "f" and len(parts) >= 4:
                face = [_parse_face_token(token, len(vertices), len(uvs)) for token in parts[1:]]
                for index in range(1, len(face) - 1):
                    tri = (face[0], face[index], face[index + 1])
                    triangles.append(
                        Triangle(
                            vertex_ids=(tri[0][0], tri[1][0], tri[2][0]),
                            uv_ids=(tri[0][1], tri[1][1], tri[2][1]),
                            material=current_material,
                        )
                    )

    materials.setdefault("default", Material(kd=np.array([0.65, 0.65, 0.65], dtype=np.float32)))
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        triangles,
        materials,
    )


def _load_textures(materials: dict[str, Material]) -> dict[Path, np.ndarray]:
    textures: dict[Path, np.ndarray] = {}
    for material in materials.values():
        if material.texture is None or material.texture in textures or not material.texture.exists():
            continue
        image = Image.open(material.texture).convert("RGB")
        textures[material.texture] = np.asarray(image, dtype=np.float32) / 255.0
    return textures


def _triangle_colors(
    triangles: list[Triangle],
    uvs: np.ndarray,
    materials: dict[str, Material],
    textures: dict[Path, np.ndarray],
) -> np.ndarray:
    colors: list[np.ndarray] = []
    for triangle in triangles:
        material = materials.get(triangle.material, materials["default"])
        color = material.kd.copy()
        if material.texture is not None and material.texture in textures and all(id_ is not None for id_ in triangle.uv_ids):
            uv = uvs[[int(id_) for id_ in triangle.uv_ids]].mean(axis=0)
            texture = textures[material.texture]
            x = int((uv[0] % 1.0) * (texture.shape[1] - 1))
            y = int(((1.0 - uv[1]) % 1.0) * (texture.shape[0] - 1))
            color = 0.25 * color + 0.75 * texture[y, x]
        colors.append(np.clip(color, 0.0, 1.0))
    return np.asarray(colors, dtype=np.float32)


def _look_at_basis(camera: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, forward


def _render_frame(
    vertices: np.ndarray,
    face_ids: np.ndarray,
    face_colors: np.ndarray,
    target: np.ndarray,
    yaw: float,
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    radius = 2.1
    camera = target + np.array([math.cos(yaw) * radius, math.sin(yaw) * radius, 0.72], dtype=np.float32)
    right, up, forward = _look_at_basis(camera, target + np.array([0.0, 0.0, 0.12], dtype=np.float32))
    relative = vertices - target
    view_x = relative @ right
    view_y = relative @ up
    depth = relative @ forward

    screen = np.empty((vertices.shape[0], 2), dtype=np.float32)
    screen[:, 0] = width * 0.5 + view_x * scale
    screen[:, 1] = height * 0.54 - view_y * scale

    frame = np.full((height, width, 3), (244, 245, 247), dtype=np.uint8)
    shadow_center = (int(width * 0.5), int(height * 0.74))
    cv2.ellipse(frame, shadow_center, (int(width * 0.27), int(height * 0.055)), 0, 0, 360, (218, 221, 226), -1)

    tri_vertices = vertices[face_ids]
    normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = normal_lengths > 1e-8
    normals[valid] /= normal_lengths[valid, None]
    light_dir = camera - target + np.array([0.0, 0.0, 0.8], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
    intensity = 0.42 + 0.58 * diffuse

    mean_depth = depth[face_ids].mean(axis=1)
    order = np.argsort(mean_depth)[::-1]
    for tri_index in order:
        points = np.rint(screen[face_ids[tri_index]]).astype(np.int32)
        if (
            points[:, 0].max() < 0
            or points[:, 0].min() >= width
            or points[:, 1].max() < 0
            or points[:, 1].min() >= height
        ):
            continue
        color = np.clip(face_colors[tri_index] * intensity[tri_index] * 255.0, 0, 255).astype(np.uint8)
        cv2.fillConvexPoly(frame, points, (int(color[2]), int(color[1]), int(color[0])), lineType=cv2.LINE_AA)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=_default_obj_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--turns", type=float, default=2.0)
    args = parser.parse_args()

    vertices, uvs, triangles, materials = _load_obj(args.obj)
    face_ids = np.asarray([triangle.vertex_ids for triangle in triangles], dtype=np.int32)
    face_colors = _triangle_colors(triangles, uvs, materials, _load_textures(materials))

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    size = bounds_max - bounds_min
    scale = min(args.width * 0.78 / max(size[0], size[1]), args.height * 0.70 / size[2])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(args.duration * args.fps))
    with imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for frame_index in range(frame_count):
            yaw = 2.0 * math.pi * args.turns * frame_index / frame_count + math.radians(35.0)
            writer.append_data(
                _render_frame(
                    vertices=vertices,
                    face_ids=face_ids,
                    face_colors=face_colors,
                    target=center,
                    yaw=yaw,
                    width=args.width,
                    height=args.height,
                    scale=scale,
                )
            )

    print(f"Wrote {args.output}")
    print(f"Frames: {frame_count}, triangles: {len(triangles)}, vertices: {len(vertices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
