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
        extents = verts.max(axis=0) - verts.min(axis=0)
        scale = np.ones(3, dtype=np.float32)
        for i in range(3):
            if extents[i] > 0:
                scale[i] = target[i] / extents[i]

        center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
        verts_scaled = (verts - center) * scale + center
        mesh_scaled = trimesh.Trimesh(vertices=verts_scaled, faces=mesh.faces, process=False)
        mesh_scaled.export(os.path.join(args.dst, fname))
