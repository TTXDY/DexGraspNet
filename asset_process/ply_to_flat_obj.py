import argparse
import json
import os

import trimesh
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True,
                        help="Input root containing <category>/<model>/models/model_normalized.ply")
    parser.add_argument("--dst_root", type=str, required=True,
                        help="Flat output directory for .obj files")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional prefix for object code (e.g., 'shapenet_')")
    args = parser.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)

    name_map = {}
    copied = 0

    categories = [d for d in os.listdir(args.src_root)
                  if os.path.isdir(os.path.join(args.src_root, d))]
    for cat in tqdm(sorted(categories), desc="categories", mininterval=1.0):
        cat_dir = os.path.join(args.src_root, cat)
        models = [d for d in os.listdir(cat_dir)
                  if os.path.isdir(os.path.join(cat_dir, d))]
        models = sorted(models)
        for i, model in enumerate(models, 1):
            src_mesh = os.path.join(cat_dir, model, "models", "model_normalized.ply")
            if not os.path.exists(src_mesh):
                continue
            obj_name = f"{args.prefix}{cat}_{i}"
            dst_mesh = os.path.join(args.dst_root, f"{obj_name}.obj")
            mesh = trimesh.load(src_mesh, force="mesh", process=False)
            mesh.export(dst_mesh)
            name_map[f"{cat}-{model}"] = obj_name
            copied += 1

    map_path = os.path.join(args.dst_root, "name_map.json")
    with open(map_path, "w") as f:
        json.dump(name_map, f, indent=2)

    print(f"Copied {copied} objects to {args.dst_root}")
    print(f"Saved {map_path}")


if __name__ == "__main__":
    main()
