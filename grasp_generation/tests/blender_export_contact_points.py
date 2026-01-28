"""
Blender script: export selected vertices to contact_points.json (link-local coords).

Usage:
1) Open Blender -> Scripting.
2) Load this script.
3) Select mesh object (active), select vertices in Edit Mode.
4) Run the script.
"""

import bpy
import bmesh
import json
import os

# === CONFIG ===
OUTPUT_PATH = "/home/jay/gesture_generation/DexGraspNet/grasp_generation/mjcf_dexhand021/contact_points.json"
LINK_NAME = None  # If None, use active object's name.
MERGE_EXISTING = True  # Merge into existing JSON if present.


def get_selected_vertices_local(obj):
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        verts = [v.co[:] for v in bm.verts if v.select]
        return verts
    # OBJECT mode: use mesh selection
    verts = [v.co[:] for v in obj.data.vertices if v.select]
    return verts


def main():
    obj = bpy.context.active_object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Select a mesh object with vertices.")

    link_name = LINK_NAME or obj.name

    # Warn if object has unapplied transforms
    if tuple(obj.scale) != (1.0, 1.0, 1.0) or tuple(obj.rotation_euler) != (0.0, 0.0, 0.0):
        print("WARNING: Object has non-identity transforms. Apply them if you want raw mesh-local coords.")

    verts = get_selected_vertices_local(obj)
    if not verts:
        raise RuntimeError("No selected vertices found.")

    # Load existing
    data = {}
    if MERGE_EXISTING and os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            data = json.load(f)

    data[link_name] = verts

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Wrote {len(verts)} points to {OUTPUT_PATH} under link '{link_name}'.")


if __name__ == "__main__":
    main()
