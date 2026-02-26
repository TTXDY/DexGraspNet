import argparse
import json
import os
import shutil


def _load_categories(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [k.lower() for k in data.keys()]
    return [str(x).lower() for x in data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True,
                        help="Input root (e.g., ShapeNetCore.v2_scaled)")
    parser.add_argument("--dst_root", type=str, required=True,
                        help="Output root for filtered categories")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category names")
    parser.add_argument("--categories_json", type=str, default=None,
                        help="JSON list or dict of categories")
    args = parser.parse_args()

    if args.categories_json:
        categories = _load_categories(args.categories_json)
    elif args.categories:
        categories = [c.strip().lower() for c in args.categories.split(",") if c.strip()]
    else:
        raise ValueError("Provide --categories or --categories_json")

    os.makedirs(args.dst_root, exist_ok=True)

    copied = 0
    missing = []
    for cat in categories:
        src_dir = os.path.join(args.src_root, cat)
        if not os.path.isdir(src_dir):
            missing.append(cat)
            continue
        dst_dir = os.path.join(args.dst_root, cat)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        copied += 1

    print(f"Copied {copied} categories to {args.dst_root}")
    if missing:
        print("Missing categories:", ", ".join(missing))
