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

from utils.hand_model import HandModel
from utils.hand_model_dexhand021 import HandModelDexHand021
from utils.object_model import ObjectModel

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']


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

    # Load results
    print("\n1. Loading results...")
    result_file = os.path.join(args.result_path, args.object_code + '.npy')
    data_dict = np.load(result_file, allow_pickle=True)[args.num]
    print(f"   ✓ Results loaded")

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

    # Extract poses (dexhand021 uses native frame)
    if 'hand_pose_raw' in data_dict:
        hand_pose = torch.tensor(data_dict['hand_pose_raw'], dtype=torch.float, device=device)
        print("   ✓ Using saved hand_pose_raw for exact pose")
    else:
        qpos = data_dict['qpos']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names], dtype=torch.float, device=device)

    if 'qpos_st' in data_dict and not args.no_init:
        qpos_st = data_dict['qpos_st']
        rot = np.array(transforms3d.euler.euler2mat(*[qpos_st[name] for name in rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose_st = torch.tensor([qpos_st[name] for name in translation_names] + rot + [qpos_st[name] for name in joint_names], dtype=torch.float, device=device)
        print(f"   ✓ Initial pose loaded")
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
        num_samples=args.object_num_samples,
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

    # Object
    object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=0.7)

    def _get_object_surface_points():
        if 'object_surface_points' in data_dict:
            pts = torch.tensor(data_dict['object_surface_points'], dtype=torch.float, device=device)
            return pts.unsqueeze(0)
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
