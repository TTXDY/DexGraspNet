import argparse
import os


def restore_object(root: str, obj: str):
    coacd_dir = os.path.join(root, obj, "coacd")
    if not os.path.isdir(coacd_dir):
        raise FileNotFoundError(f"Missing coacd dir: {coacd_dir}")
    restored = 0
    for fname in os.listdir(coacd_dir):
        if fname.endswith(".bak"):
            bak = os.path.join(coacd_dir, fname)
            orig = os.path.join(coacd_dir, fname[:-4])
            os.replace(bak, orig)
            restored += 1
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshdata_root", required=True, type=str,
                        help="Root meshdata directory")
    parser.add_argument("--object_code", required=True, type=str,
                        help="Object folder name (e.g., basket_65)")
    args = parser.parse_args()

    n = restore_object(args.meshdata_root, args.object_code)
    print(f"Restored {n} meshes for {args.object_code}")


if __name__ == "__main__":
    main()
