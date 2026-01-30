"""
Render a grid of DexHand021 grasps to an interactive Plotly HTML.

Example:
  /home/jay/anaconda3/envs/dexgraspnet-dexhand/bin/python scripts/render_grasp_grid.py \
    --result_dir ../data/experiments/dexhand021_grasping/results \
    --objects "banana,cube1,cuboid1" --grasps 6 --output dexhand_grid.html
"""

import argparse
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
parent_dir = os.path.dirname(script_dir)
if os.path.exists(parent_dir):
    os.chdir(parent_dir)
    sys.path.insert(0, parent_dir)

import numpy as np
import plotly.graph_objects as go
import torch
import transforms3d
import trimesh as tm

from utils.hand_model_dexhand021 import HandModelDexHand021


translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]


def _parse_list(text):
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    if s.startswith("["):
        return json.loads(s)
    return [t.strip() for t in s.split(",") if t.strip()]


def _parse_contact_links(text):
    if text is None:
        return None
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace("\"", "").replace("'", "")
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    return tokens or None


def _build_hand_pose(data_dict, device):
    if "hand_pose_raw" in data_dict:
        return torch.tensor(data_dict["hand_pose_raw"], dtype=torch.float, device=device)
    qpos = data_dict["qpos"]
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()

    if "controls" in data_dict:
        controls_entry = data_dict["controls"]
        if isinstance(controls_entry, dict):
            control_names = [
                "ctrl_thumb_spread",
                "ctrl_thumb_mcp",
                "ctrl_thumb_dip",
                "ctrl_finger_spread",
                "ctrl_index_mcp",
                "ctrl_index_dip",
                "ctrl_middle_mcp",
                "ctrl_middle_dip",
                "ctrl_ring_mcp",
                "ctrl_ring_dip",
                "ctrl_pinky_mcp",
                "ctrl_pinky_dip",
            ]
            controls = [controls_entry[name] for name in control_names]
        else:
            controls = list(controls_entry)
    else:
        def get(name, default=0.0):
            return qpos.get(name, default)
        if all(name in qpos for name in ["r_f_joint2_1", "r_f_joint4_1", "r_f_joint5_1"]):
            spread = (get("r_f_joint2_1") + get("r_f_joint4_1") + 0.5 * get("r_f_joint5_1")) / 3.0
        else:
            spread = get("r_f_joint2_1")
        controls = [
            get("r_f_joint1_1"),
            get("r_f_joint1_2"),
            get("r_f_joint1_3"),
            spread,
            get("r_f_joint2_2"),
            get("r_f_joint2_3"),
            get("r_f_joint3_2"),
            get("r_f_joint3_3"),
            get("r_f_joint4_2"),
            get("r_f_joint4_3"),
            get("r_f_joint5_2"),
            get("r_f_joint5_3"),
        ]

    return torch.tensor([qpos[name] for name in translation_names] + rot + controls, dtype=torch.float, device=device)


def _load_object_mesh(mesh_root, object_code):
    candidates = [
        os.path.join(mesh_root, object_code, "coacd", "decomposed.obj"),
        os.path.join(mesh_root, object_code, "coacd", "coacd.obj"),
        os.path.join(mesh_root, object_code, "decomposed.obj"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return tm.load(path, force="mesh", process=True)
    raise FileNotFoundError(f"No mesh found for {object_code} under {mesh_root}")


def _make_pose(tx, ty, tz):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _lerp(a, b, t):
    return a + (b - a) * t


def _interpolate_colorscale(colorscale, t):
    t = float(max(0.0, min(1.0, t)))
    for i in range(len(colorscale) - 1):
        t0, c0 = colorscale[i]
        t1, c1 = colorscale[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return c1
            u = (t - t0) / (t1 - t0)
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            r = int(round(_lerp(r0, r1, u)))
            g = int(round(_lerp(g0, g1, u)))
            b = int(round(_lerp(b0, b1, u)))
            return _rgb_to_hex((r, g, b))
    return colorscale[-1][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="../data/experiments/dexhand021_grasping/results")
    parser.add_argument("--objects", type=str, required=True, help="Comma list or JSON list of object codes")
    parser.add_argument("--grasps", type=int, default=6, help="Number of grasps per object")
    parser.add_argument("--start_index", type=int, default=0, help="Start index within each npy")
    parser.add_argument("--contact_links", default=None, type=str,
                        help="Expected contact link preset id or tokens to filter results.")
    parser.add_argument("--contact_links_file", default="../data/contact_link.json", type=str,
                        help="JSON mapping of contact link presets (e.g., {\"01\": [...], \"02\": [...]})")
    parser.add_argument("--mesh_root", type=str, default="../data/meshdata")
    parser.add_argument("--col_stride", type=float, default=0.35, help="Grid spacing along X")
    parser.add_argument("--row_stride", type=float, default=0.35, help="Grid spacing along Y")
    parser.add_argument("--output", type=str, default="dexhand_grid.html")
    parser.add_argument("--hand_opacity", type=float, default=1.0)
    parser.add_argument("--object_opacity", type=float, default=1.0)
    parser.add_argument("--color_by_grasp", action="store_true", default=True,
                        help="Color objects by grasp index (0..grasps-1). Enabled by default.")
    parser.add_argument("--color_by_object", action="store_true",
                        help="Color objects by object (override grasp colors).")
    args = parser.parse_args()
    if args.color_by_object:
        args.color_by_grasp = False

    objects = _parse_list(args.objects)
    if not objects:
        raise ValueError("--objects is required and cannot be empty")

    device = "cpu"
    hand_model = HandModelDexHand021(
        mjcf_path="mjcf_dexhand021/dexhand021_right_simplified_floating_cleaned.xml",
        mesh_path="mjcf_dexhand021/meshes_simplified",
        contact_points_path="mjcf_dexhand021/contact_points.json",
        penetration_points_path="mjcf_dexhand021/penetration_points.json",
        device=device,
    )

    palette = [
        "#8a6cff",
        "#c84cff",
        "#ff5a5a",
        "#ff9f43",
        "#f3c623",
        "#6dd47e",
        "#20b2aa",
        "#3aa0ff",
        "#7f8c8d",
    ]

    grasp_colorscale = [
        [0.0, "#6dd47e"],
        [0.25, "#f3c623"],
        [0.5, "#ff9f43"],
        [0.75, "#ff5a5a"],
        [1.0, "#c84cff"],
    ]

    traces = []

    for obj_idx, object_code in enumerate(objects):
        result_path = os.path.join(args.result_dir, object_code + ".npy")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Result file not found: {result_path}")
        data_arr = np.load(result_path, allow_pickle=True)

        if args.contact_links is not None:
            try:
                with open(args.contact_links_file, "r") as f:
                    contact_links_map = json.load(f)
            except FileNotFoundError:
                contact_links_map = {}
            requested = args.contact_links.strip()
            if requested.lower() == "all":
                expected_tokens = None
                expected_id = "all"
                expected_ids = None
            elif requested in contact_links_map:
                expected_tokens = contact_links_map[requested]
                expected_id = requested
                expected_ids = None
            else:
                tokens = _parse_contact_links(requested)
                if tokens and contact_links_map and all(t in contact_links_map for t in tokens):
                    expected_ids = tokens
                    expected_tokens = None
                    expected_id = "multi"
                else:
                    expected_tokens = tokens
                    expected_id = "custom"
                    expected_ids = None
            if expected_id == "all":
                match_indices = list(range(len(data_arr)))
            elif expected_id == "multi" and expected_ids:
                id_to_indices = {eid: [] for eid in expected_ids}
                for i, d in enumerate(data_arr):
                    cid = d.get("contact_links_id", None)
                    if cid in id_to_indices:
                        id_to_indices[cid].append(i)
                per_id = max(args.grasps // len(expected_ids), 1)
                remainder = max(args.grasps - per_id * len(expected_ids), 0)
                match_indices = []
                for eid in expected_ids:
                    match_indices.extend(id_to_indices[eid][:per_id])
                if remainder > 0:
                    for eid in expected_ids:
                        if remainder <= 0:
                            break
                        extra_idx = id_to_indices[eid][per_id:per_id + 1]
                        if extra_idx:
                            match_indices.extend(extra_idx)
                            remainder -= 1
            else:
                match_indices = []
                for i, d in enumerate(data_arr):
                    if d.get("contact_links_id", None) == expected_id:
                        match_indices.append(i)
                    elif expected_id == "custom" and isinstance(expected_tokens, list):
                        if d.get("contact_links_tokens", None) == expected_tokens:
                            match_indices.append(i)
            if not match_indices:
                print(f"[{object_code}] no results match contact_links={requested}; using full list.")
            else:
                data_arr = data_arr[match_indices]

        mesh = _load_object_mesh(args.mesh_root, object_code)
        for j in range(args.grasps):
            idx = args.start_index + j
            if idx >= len(data_arr):
                break
            data_dict = data_arr[idx]
            scale = float(data_dict["scale"])

            tx = j * args.col_stride
            ty = -obj_idx * args.row_stride
            pose = _make_pose(tx, ty, 0.0)

            hand_pose = _build_hand_pose(data_dict, device)
            hand_model.set_parameters(hand_pose.unsqueeze(0))
            hand_traces = hand_model.get_plotly_data(
                i=0, opacity=args.hand_opacity, color="#b0b0b0", with_contact_points=False, pose=pose
            )
            traces.extend(hand_traces)

            if args.color_by_grasp:
                denom = max(args.grasps - 1, 1)
                t_color = j / denom
                color = _interpolate_colorscale(grasp_colorscale, t_color)
            else:
                color = palette[obj_idx % len(palette)]

            v = mesh.vertices * scale
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            f = mesh.faces
            traces.append(
                go.Mesh3d(
                    x=v[:, 0],
                    y=v[:, 1],
                    z=v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=args.object_opacity,
                    flatshading=True,
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="#2b2b2b",
        plot_bgcolor="#2b2b2b",
    )
    fig.write_html(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
