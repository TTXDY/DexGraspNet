"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
Supports both Shadow Hand and DexHand021 models
"""

import os
import sys

# Get the script's directory and parent directory (grasp_generation/)
script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
parent_dir = os.path.dirname(script_dir)

# Change to parent directory and add to path
if os.path.exists(parent_dir):
    os.chdir(parent_dir)
    sys.path.insert(0, parent_dir)

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
import json

from utils.hand_model import HandModel
from utils.hand_model_dexhand021 import HandModelDexHand021
from utils.object_model import ObjectModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

def _parse_contact_links(text):
    if text is None:
        return None
    s = text.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    s = s.replace('"', '').replace("'", '')
    tokens = [t.strip() for t in s.split(',') if t.strip()]
    return tokens or None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand_model_type', type=str, default='shadow_hand', choices=['shadow_hand', 'dexhand021'])
    parser.add_argument('--object_code', type=str, default='sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2')
    parser.add_argument('--num', type=int, default=0, help='Grasp index (0=best)')
    parser.add_argument('--result_path', type=str, default='../data/dataset')
    parser.add_argument('--show_contact_points', action='store_true', help='Show contact points in red')
    parser.add_argument('--no_init', action='store_true', help='Hide initial pose (show only optimized pose)')
    parser.add_argument('--object_num_samples', type=int, default=2000,
                        help='Number of object surface points used for E_pen debugging.')
    parser.add_argument('--contact_links', default=None, type=str,
                        help='Expected contact link preset id or tokens to verify against saved result.')
    parser.add_argument('--contact_links_file', default='../data/contact_link.json', type=str,
                        help='JSON mapping of contact link presets (e.g., {"01": [...], "02": [...]})')
    parser.add_argument('--show_collision_capsules', action='store_true',
                        help='Show collision capsules (DexHand021 only).')
    parser.add_argument('--show_penetrating_points', action='store_true',
                        help='Show object surface points that fall inside collision capsules (black).')
    parser.add_argument('--max_pen_points', type=int, default=2000,
                        help='Max number of penetrating points to render.')
    args = parser.parse_args()

    device = 'cpu'

    print("=" * 80)
    print(f"VISUALIZING {args.hand_model_type.upper()} GRASP RESULT")
    print("=" * 80)
    print(f"Object: {args.object_code}")
    print(f"Grasp index: {args.num}")
    print(f"Result path: {args.result_path}")

    # Visualization-only global offset (object + hand)
    vis_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device)

    # Load results
    print("\n1. Loading results...")
    result_file = os.path.join(args.result_path, args.object_code + '.npy')
    data_arr = np.load(result_file, allow_pickle=True)
    data_idx = args.num
    print(f"   ✓ Results loaded")
    # Optional contact link check
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
        elif requested in contact_links_map:
            expected_tokens = contact_links_map[requested]
            expected_id = requested
        else:
            expected_tokens = _parse_contact_links(requested)
            expected_id = "custom"
        # Remap args.num to the subset that matches the requested contact links.
        if expected_id is None or expected_id == "all":
            match_indices = list(range(len(data_arr)))
        else:
            match_indices = []
            for i, d in enumerate(data_arr):
                if d.get('contact_links_id', None) == expected_id:
                    match_indices.append(i)
                elif expected_id == "custom" and isinstance(expected_tokens, list):
                    if d.get('contact_links_tokens', None) == expected_tokens:
                        match_indices.append(i)
        if not match_indices:
            print(f"   ⚠ No results match contact_links={requested}; using global index {data_idx}.")
        else:
            if data_idx < 0 or data_idx >= len(match_indices):
                raise ValueError(f"Requested num {data_idx} out of range for contact_links={requested} (0..{len(match_indices)-1}).")
            mapped_idx = match_indices[data_idx]
            print(f"   ✓ contact_links subset size: {len(match_indices)}; mapped index {data_idx} -> {mapped_idx}")
            data_idx = mapped_idx
        data_dict = data_arr[data_idx]
        saved_id = data_dict.get('contact_links_id', None)
        saved_tokens = data_dict.get('contact_links_tokens', None)
        if saved_id != expected_id:
            print(f"   ⚠ contact_links_id mismatch: saved={saved_id}, requested={expected_id}")
        if isinstance(expected_tokens, list) and saved_tokens is not None and saved_tokens != expected_tokens:
            print(f"   ⚠ contact_links_tokens mismatch: saved={saved_tokens}, requested={expected_tokens}")
    else:
        data_dict = data_arr[data_idx]

    # Load hand model
    print("\n2. Loading HandModel...")
    if args.hand_model_type == 'dexhand021':
        hand_model = HandModelDexHand021(
            mjcf_path='mjcf_dexhand021/dexhand021_right_simplified_floating_cleaned.xml',
            mesh_path='mjcf_dexhand021/meshes',
            contact_points_path='mjcf_dexhand021/contact_points.json',
            penetration_points_path='mjcf_dexhand021/penetration_points.json',
            device=device
        )
    else:
        hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            device=device
        )
    print(f"   ✓ HandModel loaded (DOF: {hand_model.n_dofs})")
    joint_names = hand_model.joints_names
    if len(joint_names) != hand_model.n_dofs:
        raise ValueError(f"Joint name count ({len(joint_names)}) does not match DOF ({hand_model.n_dofs}).")
    if args.hand_model_type == 'dexhand021':
        control_names = joint_names

    # Extract poses (dexhand021 uses native frame)
    if 'hand_pose_raw' in data_dict:
        hand_pose = torch.tensor(data_dict['hand_pose_raw'], dtype=torch.float, device=device)
        print("   ✓ Using saved hand_pose_raw for exact pose")
        try:
            _final_t_raw = hand_pose[:3].detach().cpu().tolist()
            _final_rot6d = hand_pose[3:9].detach().cpu().tolist()
            print(f"   - Hand final translation (hand_pose_raw): {_final_t_raw}")
            print(f"   - Hand final rotation (rot6d): {_final_rot6d}")
            if 'intrinsic_euler_hand_pose_3_3_12' in data_dict and data_dict['intrinsic_euler_hand_pose_3_3_12'] is not None:
                _hp = data_dict['intrinsic_euler_hand_pose_3_3_12']
                if len(_hp) >= 6:
                    _final_euler = _hp[3:6]
                    print(f"   - Hand final rotation (intrinsic euler xyz): {_final_euler}")
        except Exception:
            pass
        hand_pose[:3] += vis_offset
        try:
            _final_t = hand_pose[:3].detach().cpu().tolist()
            print(f"   - Hand final translation (hand_pose_raw, offset): {_final_t}")
        except Exception:
            pass
        try:
            _final_t = hand_pose[:3].detach().cpu().tolist()
            print(f"   - Hand final translation (hand_pose_raw): {_final_t}")
        except Exception:
            pass
    else:
        qpos = data_dict['qpos']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        if args.hand_model_type == 'dexhand021':
            if 'controls' in data_dict:
                controls_entry = data_dict['controls']
                if isinstance(controls_entry, dict):
                    controls = [controls_entry[name] for name in control_names]
                else:
                    controls = list(controls_entry)
            elif all(name in qpos for name in control_names):
                controls = [qpos[name] for name in control_names]
            else:
                def get(name, default=0.0):
                    return qpos.get(name, default)
                if all(name in qpos for name in ['r_f_joint2_1', 'r_f_joint4_1', 'r_f_joint5_1']):
                    spread = (get('r_f_joint2_1') + get('r_f_joint4_1') + 0.5 * get('r_f_joint5_1')) / 3.0
                else:
                    spread = get('r_f_joint2_1')
                controls = [
                    get('r_f_joint1_1'),
                    get('r_f_joint1_2'),
                    get('r_f_joint1_3'),
                    spread,
                    get('r_f_joint2_2'),
                    get('r_f_joint2_3'),
                    get('r_f_joint3_2'),
                    get('r_f_joint3_3'),
                    get('r_f_joint4_2'),
                    get('r_f_joint4_3'),
                    get('r_f_joint5_2'),
                    get('r_f_joint5_3'),
                ]
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + controls, dtype=torch.float, device=device)
            hand_pose[:3] += vis_offset
        else:
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)
            hand_pose[:3] += vis_offset

    if 'qpos_st' in data_dict and not args.no_init:
        qpos_st = data_dict['qpos_st']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        if args.hand_model_type == 'dexhand021':
            if 'controls_st' in data_dict:
                controls_entry = data_dict['controls_st']
                if isinstance(controls_entry, dict):
                    controls = [controls_entry[name] for name in control_names]
                else:
                    controls = list(controls_entry)
            elif all(name in qpos_st for name in control_names):
                controls = [qpos_st[name] for name in control_names]
            else:
                def get(name, default=0.0):
                    return qpos_st.get(name, default)
                if all(name in qpos_st for name in ['r_f_joint2_1', 'r_f_joint4_1', 'r_f_joint5_1']):
                    spread = (get('r_f_joint2_1') + get('r_f_joint4_1') + 0.5 * get('r_f_joint5_1')) / 3.0
                else:
                    spread = get('r_f_joint2_1')
                controls = [
                    get('r_f_joint1_1'),
                    get('r_f_joint1_2'),
                    get('r_f_joint1_3'),
                    spread,
                    get('r_f_joint2_2'),
                    get('r_f_joint2_3'),
                    get('r_f_joint3_2'),
                    get('r_f_joint3_3'),
                    get('r_f_joint4_2'),
                    get('r_f_joint4_3'),
                    get('r_f_joint5_2'),
                    get('r_f_joint5_3'),
                ]
            hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + controls, dtype=torch.float, device=device)
            hand_pose_st[:3] += vis_offset
        else:
            hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + [qpos_st[name] for name in joint_names], dtype=torch.float, device=device)
            hand_pose_st[:3] += vis_offset
        print(f"   ✓ Initial pose loaded")
        try:
            _init_t = [qpos_st[name] for name in translation_names]
            _init_rot = [qpos_st[name] for name in rot_names]
            print(f"   - Hand initial translation (qpos_st): {_init_t}")
            print(f"   - Hand initial rotation (euler xyz): {_init_rot}")
        except Exception:
            pass
    else:
        hand_pose_st = None
        if args.no_init:
            print("   ✓ Initial pose hidden (--no_init)")
        else:
            print(f"   ⚠ No initial pose found")

    # Load object model
    print("\n3. Loading ObjectModel...")
    object_model = ObjectModel(
        data_root_path='../data/meshdata',
        batch_size_each=1,
        num_surface_samples=args.object_num_samples,
        device=device
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = torch.tensor(
        data_dict['scale'], dtype=torch.float, device=device
    ).reshape(1, 1)
    print(f"   ✓ ObjectModel loaded")
    print(f"   - Object scale: {data_dict['scale']:.4f}")
    if 'object_surface_points' in data_dict:
        print("   ✓ Using saved object surface points from results")
        try:
            import numpy as _np
            _pts = _np.asarray(data_dict['object_surface_points'], dtype=_np.float32) + vis_offset.detach().cpu().numpy()
            if _pts.size > 0:
                _obj_center = _pts.mean(axis=0)
                print(f"   - Object center (from surface points): {_obj_center.tolist()}")
                _mins = _pts.min(axis=0)
                _maxs = _pts.max(axis=0)
                _lens = (_maxs - _mins).tolist()
                print(f"   - Object xyz lengths (from surface points): {_lens}")
        except Exception:
            pass
    if 'init_choice_index' in data_dict:
        print(f"   - Init choice index: {data_dict['init_choice_index']}")

    # Create visualization
    print("\n4. Creating visualization...")
    if hasattr(hand_model, "output_align"):
        align = hand_model.output_align.detach().cpu()
        if not torch.allclose(align, torch.eye(3), atol=1e-6):
            print("   ⚠ output_align is non-identity; collision capsules are aligned for display, but E_pen does not use output_align.")

    # If results don't include contact_point_indices, disable contact point display.
    if args.show_contact_points and 'contact_point_indices' not in data_dict:
        print("   ⚠ contact_point_indices not found in result; rerun main.py to save them.")
        args.show_contact_points = False

    # Initial pose (transparent)
    if hand_pose_st is not None:
        if args.show_contact_points and 'contact_point_indices' in data_dict:
            cp_idx = torch.tensor(data_dict['contact_point_indices'], dtype=torch.long, device=device).unsqueeze(0)
            hand_model.set_parameters(hand_pose_st.unsqueeze(0), cp_idx)
        else:
            hand_model.set_parameters(hand_pose_st.unsqueeze(0))
        hand_st_plotly = hand_model.get_plotly_data(
            i=0, opacity=0.3, color='lightblue', with_contact_points=args.show_contact_points
        )
    else:
        hand_st_plotly = []

    # Optimized pose (opaque)
    if args.show_contact_points and 'contact_point_indices' in data_dict:
        cp_idx = torch.tensor(data_dict['contact_point_indices'], dtype=torch.long, device=device).unsqueeze(0)
        hand_model.set_parameters(hand_pose.unsqueeze(0), cp_idx)
        # Print link names for selected contact points
        try:
            link_indices = hand_model.global_index_to_link_index[cp_idx[0]].detach().cpu().tolist()
            idx_to_name = {v: k for k, v in hand_model.link_name_to_link_index.items()}
            link_names = [idx_to_name.get(i, f'idx_{i}') for i in link_indices]
            print(f"   - Contact links: {link_names}")
        except Exception:
            pass
    else:
        hand_model.set_parameters(hand_pose.unsqueeze(0))
    hand_en_plotly = hand_model.get_plotly_data(
        i=0, opacity=1.0, color='lightblue', with_contact_points=args.show_contact_points
    )

    # Collision capsules (DexHand021 only)
    collision_capsules_plotly = []
    if args.show_collision_capsules and args.hand_model_type == 'dexhand021':
        import trimesh as tm
        for p0, p1, radius in hand_model._build_collision_capsules_world():
            if p0.dim() == 2:
                p0 = p0[0]
                p1 = p1[0]
            if hasattr(hand_model, "output_align"):
                p0 = p0 @ hand_model.output_align.T
                p1 = p1 @ hand_model.output_align.T
            cyl = tm.creation.cylinder(radius=radius, segment=[p0.detach().cpu().numpy(), p1.detach().cpu().numpy()])
            v = cyl.vertices
            f = cyl.faces
            collision_capsules_plotly.append(
                go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='yellow', opacity=0.35)
            )
        # Palm box
        palm_box = hand_model._build_palm_box_world_batch()
        if palm_box is not None:
            center, axes, extents = palm_box
            center = center[0].detach().cpu().numpy()
            axes = axes[0].detach().cpu().numpy()
            extents = extents[0].detach().cpu().numpy()
            sx, sy, sz = extents
            corners = np.array([
                [-sx, -sy, -sz],
                [ sx, -sy, -sz],
                [ sx,  sy, -sz],
                [-sx,  sy, -sz],
                [-sx, -sy,  sz],
                [ sx, -sy,  sz],
                [ sx,  sy,  sz],
                [-sx,  sy,  sz],
            ])
            verts = corners @ axes.T + center
            faces = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5],
                [0, 3, 7], [0, 7, 4],
            ])
            collision_capsules_plotly.append(
                go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                          color='orange', opacity=0.4)
            )

    # Object
    _pose = np.eye(4, dtype=np.float32)
    _pose[:3, 3] = vis_offset.detach().cpu().numpy()
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=0.7, pose=_pose)

    def _get_object_surface_points():
        if 'object_surface_points' in data_dict:
            pts = torch.tensor(data_dict['object_surface_points'], dtype=torch.float, device=device)
            return (pts + vis_offset).unsqueeze(0)
        object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
        return object_model.surface_points_tensor * object_scale

    # Combine all
    pen_points_plotly = []
    if args.show_penetrating_points:
        with torch.no_grad():
            object_surface_points = _get_object_surface_points()
            distances = hand_model.cal_distance(object_surface_points)
            pen_mask = distances > 0
            pen_points = object_surface_points[pen_mask]
            if pen_points.numel() > 0:
                if pen_points.shape[0] > args.max_pen_points:
                    idx = torch.randperm(pen_points.shape[0])[:args.max_pen_points]
                    pen_points = pen_points[idx]
                pts = pen_points.detach().cpu().numpy()
                pen_points_plotly = [go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers', marker=dict(color='black', size=2),
                    name='penetrating_points'
                )]

    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly + collision_capsules_plotly + pen_points_plotly)

    # Add energy information if available
    if 'energy' in data_dict:
        E_fc = data_dict['E_fc']
        E_dis = data_dict['E_dis']
        E_pen = data_dict['E_pen']
        E_spen = data_dict['E_spen']
        E_joints = data_dict['E_joints']
        result_text = (f'Index {args.num}  E_fc {E_fc:.6f}  E_dis {E_dis:.8f}  '
                       f'E_pen {E_pen:.8f}  E_spen {E_spen:.8f}  E_joints {E_joints:.6f}')
        fig.add_annotation(text=result_text, x=0.5, y=0.1, xref='paper', yref='paper')
        print(f"   - Energy: E_fc={E_fc:.6f}, E_dis={E_dis:.8f}, E_pen={E_pen:.8f}")
    else:
        print("   - Energy fields not found in result.")

    # Debug E_pen with current object_num_samples and current pose
    with torch.no_grad():
        object_surface_points = _get_object_surface_points()
        distances = hand_model.cal_distance(object_surface_points)
        pen_mask = distances > 0
        pen_count = int(pen_mask.sum().item())
        max_pen = float(distances.max().item())
        sum_pen = float(distances.clamp_min(0).sum().item())
        print(f"   - E_pen debug (num_samples={args.object_num_samples}): "
              f"penetrating_points={pen_count}, max_pen={max_pen:.8f}, sum_pen={sum_pen:.8f}")
        if 'E_pen' in data_dict:
            diff = float(sum_pen - float(data_dict['E_pen']))
            print(f"   - E_pen consistency: saved={float(data_dict['E_pen']):.8f}, "
                  f"recomputed={sum_pen:.8f}, diff={diff:.8f}")
        if 'E_pen_recomputed' in data_dict:
            diff_saved = float(sum_pen - float(data_dict['E_pen_recomputed']))
            print(f"   - E_pen consistency (saved recomputed): saved_recomputed={float(data_dict['E_pen_recomputed']):.8f}, "
                  f"recomputed_now={sum_pen:.8f}, diff={diff_saved:.8f}")

    fig.update_layout(
        scene_aspectmode='data',
        title=f'{args.hand_model_type} Grasp Result - {args.object_code} (index {args.num})'
    )

    print(f"   ✓ Opening visualization in browser...")
    fig.show()

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
