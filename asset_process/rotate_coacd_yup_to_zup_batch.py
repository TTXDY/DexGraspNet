import argparse
import os

import numpy as np
import trimesh
from tqdm import tqdm


def rotate_mesh_yup_to_zup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Rotate +90 deg about X to convert Y-up to Z-up
    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    verts = verts @ rot.T
    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)


def rotate_object(coacd_dir: str, backup: bool):
    obj_files = [f for f in os.listdir(coacd_dir) if f.endswith(".obj")]
    if not obj_files:
        return 0
    for fname in obj_files:
        path = os.path.join(coacd_dir, fname)
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh_rot = rotate_mesh_yup_to_zup(mesh)
        if backup:
            os.replace(path, path + ".bak")
        mesh_rot.export(path)
    return len(obj_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshdata_root", required=True, type=str,
                        help="Root meshdata directory")
    parser.add_argument("--backup", action="store_true",
                        help="Write .bak copies before overwriting")
    args = parser.parse_args()

    root = args.meshdata_root
    objects = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    total = 0
    skipped = 0
    for obj in tqdm(sorted(objects), desc="rotate", mininterval=1.0):
        coacd_dir = os.path.join(root, obj, "coacd")
        if not os.path.isdir(coacd_dir):
            skipped += 1
            continue
        total += rotate_object(coacd_dir, args.backup)
    print(f"Done. Rotated meshes: {total}. Skipped objects: {skipped}.")


if __name__ == "__main__":
    main()
