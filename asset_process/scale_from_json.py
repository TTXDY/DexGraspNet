import argparse
import json
import os

import numpy as np
import trimesh
from tqdm import tqdm


def _load_size_map(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Normalized models directory")
    parser.add_argument("--dst", type=str, required=True, help="Output directory for scaled models")
    parser.add_argument("--size_json", type=str, default=None,
                        help="Path to object_real_size.json (default: <src>/object_real_size.json)")
    args = parser.parse_args()

    size_json = args.size_json or os.path.join(args.src, "object_real_size.json")
    size_map = _load_size_map(size_json)

    os.makedirs(args.dst, exist_ok=True)

    for fname in tqdm(os.listdir(args.src)):
        if not fname.endswith(".obj"):
            continue
        stem = os.path.splitext(fname)[0]
        if stem not in size_map:
            continue
        size_entry = size_map[stem]
        if "size" not in size_entry or len(size_entry["size"]) != 3:
            continue
        target = np.array(size_entry["size"], dtype=np.float32)

        mesh = trimesh.load(os.path.join(args.src, fname), force="mesh", process=False)
        if mesh.vertices is None or len(mesh.vertices) == 0:
            continue
        verts = np.array(mesh.vertices, dtype=np.float32)

        stem_lower = stem.lower()
        if "sphere" in stem_lower:
            max_norm = np.max(np.linalg.norm(verts, axis=1))
            if max_norm <= 0:
                continue
            scale = (target[0] * 0.5) / max_norm
            verts_scaled = verts * scale
        elif "cylinder" in stem_lower:
            xy = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
            max_r = np.max(xy)
            z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
            height = z_max - z_min
            if max_r <= 0 or height <= 0:
                continue
            scale_r = (target[0] * 0.5) / max_r
            scale_z = target[2] / height
            verts_scaled = verts.copy()
            verts_scaled[:, 0] *= scale_r
            verts_scaled[:, 1] *= scale_r
            verts_scaled[:, 2] *= scale_z
        elif "cube" in stem_lower or "cuboid" in stem_lower:
            center = verts.mean(axis=0)
            v0 = verts - center
            half = np.max(np.abs(v0), axis=0)
            if np.any(half <= 0):
                continue
            scale = (target * 0.5) / half
            verts_scaled = v0 * scale + center
        else:
            # Fallback: no scaling for unknown types
            verts_scaled = verts

        mesh_scaled = trimesh.Trimesh(vertices=verts_scaled, faces=mesh.faces, process=False)
        mesh_scaled.export(os.path.join(args.dst, fname))
