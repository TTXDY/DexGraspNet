import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from tqdm import tqdm


def _parse_urdf_geometry(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    visual = root.find(".//visual/geometry")
    if visual is None:
        return None
    box = visual.find("box")
    if box is not None:
        size = [float(v) for v in box.get("size").split()]
        return {"type": "box", "size": size}
    cyl = visual.find("cylinder")
    if cyl is not None:
        return {
            "type": "cylinder",
            "radius": float(cyl.get("radius")),
            "length": float(cyl.get("length")),
        }
    sphere = visual.find("sphere")
    if sphere is not None:
        return {"type": "sphere", "radius": float(sphere.get("radius"))}
    return None


def _compute_scale(geom, extents):
    if geom["type"] == "box":
        target = np.array(geom["size"], dtype=np.float32)
        scale = target / extents
        return scale
    if geom["type"] == "cylinder":
        target_d = 2.0 * geom["radius"]
        target_l = geom["length"]
        xy = np.mean(extents[:2])
        sx = target_d / xy if xy > 0 else 1.0
        sz = target_l / extents[2] if extents[2] > 0 else 1.0
        return np.array([sx, sx, sz], dtype=np.float32)
    if geom["type"] == "sphere":
        target_d = 2.0 * geom["radius"]
        s = target_d / np.max(extents) if np.max(extents) > 0 else 1.0
        return np.array([s, s, s], dtype=np.float32)
    return np.array([1.0, 1.0, 1.0], dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Normalized models directory")
    parser.add_argument("--dst", type=str, required=True, help="Output directory for real-scale models")
    parser.add_argument("--urdf_dir", type=str, required=True, help="URDF directory with real sizes")
    parser.add_argument("--write_map", action="store_true", help="Write real_size_map.json in dst")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    size_map = {}
    for fname in tqdm(os.listdir(args.src)):
        if not fname.endswith(".obj"):
            continue
        stem = os.path.splitext(fname)[0]
        urdf_path = os.path.join(args.urdf_dir, f"{stem}.urdf")
        if not os.path.exists(urdf_path):
            continue
        geom = _parse_urdf_geometry(urdf_path)
        if geom is None:
            continue
        mesh = trimesh.load(os.path.join(args.src, fname), force="mesh", process=False)
        if mesh.vertices is None or len(mesh.vertices) == 0:
            continue
        verts = np.array(mesh.vertices, dtype=np.float32)
        extents = verts.max(axis=0) - verts.min(axis=0)
        scale = _compute_scale(geom, extents)
        verts_scaled = verts * scale
        mesh_scaled = trimesh.Trimesh(vertices=verts_scaled, faces=mesh.faces, process=False)
        mesh_scaled.export(os.path.join(args.dst, fname))
        size_map[stem] = {"geom": geom, "scale": scale.tolist()}

    if args.write_map:
        with open(os.path.join(args.dst, "real_size_map.json"), "w") as f:
            json.dump(size_map, f, indent=2)
