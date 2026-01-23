#!/usr/bin/env python3
"""
Alternative manifold conversion using trimesh
This script repairs and converts meshes to manifold format
"""
import os
import trimesh
from tqdm import tqdm
import argparse

def process_mesh(input_path, output_path):
    """Load, repair, and save mesh"""
    try:
        # Load mesh
        mesh = trimesh.load(input_path, force='mesh', process=True)

        # Repair mesh - fill holes, remove duplicates, etc.
        if not mesh.is_watertight:
            # Try to fix the mesh
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_inversion(mesh)
            mesh.fill_holes()

        # Export as OBJ
        mesh.export(output_path)
        return True, mesh.is_watertight
    except Exception as e:
        return False, False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    files = [f for f in os.listdir(args.src) if f.endswith('.stl')]

    success_count = 0
    watertight_count = 0

    for filename in tqdm(files):
        input_path = os.path.join(args.src, filename)
        # Change extension to .obj
        output_filename = filename.replace('.stl', '.obj')
        output_path = os.path.join(args.dst, output_filename)

        success, is_watertight = process_mesh(input_path, output_path)

        if success:
            success_count += 1
            if is_watertight:
                watertight_count += 1
                print(f"✓ {filename} -> watertight")
            else:
                print(f"⚠ {filename} -> not watertight (but converted)")
        else:
            print(f"✗ {filename} -> failed")

    print(f"\nResults: {success_count}/{len(files)} converted, {watertight_count} watertight")
