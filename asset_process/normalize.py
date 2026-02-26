import argparse
import os

import numpy as np
import tqdm
import trimesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument("--require_watertight", action="store_true", default=True,
                        help="Require watertight meshes (default: True).")
    parser.add_argument("--no_require_watertight", action="store_false",
                        dest="require_watertight",
                        help="Allow non-watertight meshes.")
    parser.add_argument("--min_volume", type=float, default=0.05,
                        help="Minimum volume threshold (default: 0.05).")
    parser.add_argument("--scale", action="store_true", default=True,
                        help="Enable size normalization (default: True).")
    parser.add_argument("--no_scale", action="store_false", dest="scale",
                        help="Center meshes only, keep original size.")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    for code in tqdm.tqdm(os.listdir(args.src)):
        src_path = os.path.join(args.src, code)
        if not os.path.isfile(src_path):
            continue

        mesh = trimesh.load(src_path, force="mesh", process=False)
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            continue

        xcenter = (np.max(verts[:, 0]) + np.min(verts[:, 0])) / 2
        ycenter = (np.max(verts[:, 1]) + np.min(verts[:, 1])) / 2
        zcenter = (np.max(verts[:, 2]) + np.min(verts[:, 2])) / 2
        verts_ = verts - np.array([xcenter, ycenter, zcenter])

        if args.scale:
            dmax = np.max(np.sqrt(np.sum(np.square(verts_), axis=1))) * 1.03
            if dmax > 0:
                verts_ = verts_ / dmax

        mesh_ = trimesh.Trimesh(vertices=verts_, faces=mesh.faces, process=False)
        if args.require_watertight and not mesh_.is_watertight:
            continue
        if mesh_.volume <= args.min_volume:
            continue
        mesh_.export(os.path.join(args.dst, code))
