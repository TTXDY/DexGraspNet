import argparse
import os

import numpy as np
import trimesh


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshdata_root", required=True, type=str,
                        help="Root meshdata directory")
    parser.add_argument("--object_code", required=True, type=str,
                        help="Object folder name (e.g., basket_65)")
    parser.add_argument("--backup", action="store_true",
                        help="Write .bak copies before overwriting")
    args = parser.parse_args()

    coacd_dir = os.path.join(args.meshdata_root, args.object_code, "coacd")
    if not os.path.isdir(coacd_dir):
        raise FileNotFoundError(f"Missing coacd dir: {coacd_dir}")

    obj_files = [f for f in os.listdir(coacd_dir) if f.endswith(".obj")]
    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in {coacd_dir}")

    for fname in obj_files:
        path = os.path.join(coacd_dir, fname)
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh_rot = rotate_mesh_yup_to_zup(mesh)
        if args.backup:
            os.replace(path, path + ".bak")
        mesh_rot.export(path)

    print(f"Rotated {len(obj_files)} meshes in {coacd_dir}")


if __name__ == "__main__":
    main()
