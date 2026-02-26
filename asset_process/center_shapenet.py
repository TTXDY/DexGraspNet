import argparse
import os

import numpy as np
import trimesh
from tqdm import tqdm


def _center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.size == 0:
        return mesh
    center = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
    verts_centered = verts - center
    return trimesh.Trimesh(vertices=verts_centered, faces=mesh.faces, process=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True,
                        help="Input ShapeNet root (e.g., /mnt/nas/.../ShapeNetCore.v2_scaled)")
    parser.add_argument("--dst_root", type=str, required=True,
                        help="Output root for centered meshes")
    args = parser.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)

    synsets = [d for d in os.listdir(args.src_root)
               if os.path.isdir(os.path.join(args.src_root, d))]
    for synset in tqdm(sorted(synsets), desc="synsets", mininterval=1.0):
        syn_dir = os.path.join(args.src_root, synset)
        models = [d for d in os.listdir(syn_dir)
                  if os.path.isdir(os.path.join(syn_dir, d))]
        for model in tqdm(models, desc=f"models/{synset}", leave=False, mininterval=1.0):
            src_mesh = os.path.join(syn_dir, model, "models", "model_normalized.ply")
            if not os.path.exists(src_mesh):
                continue
            mesh = trimesh.load(src_mesh, force="mesh", process=False)
            mesh_centered = _center_mesh(mesh)
            dst_dir = os.path.join(args.dst_root, synset, model, "models")
            os.makedirs(dst_dir, exist_ok=True)
            dst_mesh = os.path.join(dst_dir, "model_normalized.ply")
            mesh_centered.export(dst_mesh)
