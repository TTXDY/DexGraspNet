import os
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', required=True, help='Dex-YCB models root (contains object folders).')
    parser.add_argument('--dst_root', required=True, help='Destination flat obj folder.')
    parser.add_argument('--obj_name', default='textured_simple.obj', help='Obj filename inside each object folder.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    args = parser.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)

    copied = 0
    skipped = 0
    for name in sorted(os.listdir(args.src_root)):
        src_obj = os.path.join(args.src_root, name, args.obj_name)
        if not os.path.isfile(src_obj):
            skipped += 1
            continue
        dst_obj = os.path.join(args.dst_root, f"{name}.obj")
        if os.path.exists(dst_obj) and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(src_obj, dst_obj)
        copied += 1

    print(f"Copied {copied} objs to {args.dst_root}")
    print(f"Skipped {skipped} (missing or existing)")


if __name__ == '__main__':
    main()
