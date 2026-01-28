"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize hand model using plotly.graph_objects
Modified: 2026.01.26 - Added support for dexhand021
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.append(parent_dir)

import argparse
import numpy as np
import torch
import trimesh as tm
import transforms3d
import plotly.graph_objects as go
from utils.hand_model import HandModel
from utils.hand_model_dexhand021 import HandModelDexHand021


torch.manual_seed(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand_model_type', type=str, default='shadow_hand',
                        choices=['shadow_hand', 'dexhand021'],
                        help='Hand model type: shadow_hand or dexhand021')
    parser.add_argument('--only_penetration_points', action='store_true',
                        help='Only show penetration keypoints (brown spheres) and labels')
    parser.add_argument('--only_boxes', action='store_true',
                        help='Only show collision boxes (ShadowHand only)')
    args = parser.parse_args()

    device = torch.device('cpu')

    # Load hand model based on type
    if args.hand_model_type == 'dexhand021':
        hand_model = HandModelDexHand021(
            mjcf_path='mjcf_dexhand021/dexhand021_right_simplified_floating_cleaned.xml',
            mesh_path='mjcf_dexhand021/meshes',
            contact_points_path='mjcf_dexhand021/contact_points.json',
            penetration_points_path='mjcf_dexhand021/penetration_points.json',
            n_surface_points=2000,
            device=device
        )
        # dexhand021 default joint angles (20 DOF)
        joint_angles = torch.tensor([
            np.deg2rad(50), 0.3, 0.3, 0.3,  # Finger 1 (thumb) - first link set to 50 deg
            0.0, 0.3, 0.3, 0.3,            # Finger 2
            0.0, 0.3, 0.3, 0.3,            # Finger 3
            0.0, 0.3, 0.3, 0.3,            # Finger 4
            0.0, 0.3, 0.3, 0.3             # Finger 5
        ], dtype=torch.float, device=device)
        print(f'Visualizing dexhand021 ({hand_model.n_dofs} DOF)')
    else:
        hand_model = HandModel(
            mjcf_path='mjcf/shadow_hand_wrist_free.xml',
            mesh_path='mjcf/meshes',
            contact_points_path='mjcf/contact_points.json',
            penetration_points_path='mjcf/penetration_points.json',
            n_surface_points=2000,
            device=device
        )
        # Shadow Hand default joint angles (22 DOF)
        joint_angles = torch.tensor([0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0], dtype=torch.float, device=device)
        print(f'Visualizing Shadow Hand ({hand_model.n_dofs} DOF)')

    if args.hand_model_type == 'dexhand021':
        # rotation = torch.tensor(transforms3d.euler.euler2mat(0, np.deg2rad(0), 0, axes='sxyz'), dtype=torch.float, device=device)
        rotation = torch.tensor(transforms3d.euler.euler2mat(np.pi, np.deg2rad(-30), np.pi / 2, axes='sxyz'), dtype=torch.float, device=device)

    else:
        rotation = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device)
    hand_pose = torch.cat([torch.tensor([0, 0, 0], dtype=torch.float, device=device), rotation.T.ravel()[:6], joint_angles])
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # info
    contact_candidates = hand_model.get_contact_candidates()[0].detach().cpu().numpy()
    penetration_keypoints = hand_model.get_penetraion_keypoints()[0].detach().cpu().numpy()
    
    print('n_contact_candidates', contact_candidates.shape[0])

    # visualize

    only_geometry = args.only_penetration_points or args.only_boxes
    hand_plotly = [] if only_geometry else hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', with_contact_points=False)
    contact_candidates_plotly = [] if only_geometry else [go.Scatter3d(x=contact_candidates[:, 0], y=contact_candidates[:, 1], z=contact_candidates[:, 2], mode='markers', marker=dict(color='white', size=2))]
    contact_labels_plotly = []
    if not only_geometry:
        for link_name in hand_model.mesh:
            pts = hand_model.mesh[link_name]['contact_candidates']
            if pts is None or pts.numel() == 0:
                continue
            pts = hand_model.current_status[link_name].transform_points(pts)
            pts = pts @ hand_model.global_rotation[0].T + hand_model.global_translation[0]
            # no output alignment
            centroid = pts.mean(dim=0).detach().cpu().numpy()
            contact_labels_plotly.append(
                go.Scatter3d(
                    x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
                    mode='text',
                    text=[link_name],
                    textfont=dict(size=9, color='black'),
                    showlegend=False
                )
            )
    penetration_keypoints_plotly = [go.Scatter3d(x=penetration_keypoints[:, 0], y=penetration_keypoints[:, 1], z=penetration_keypoints[:, 2], mode='markers', marker=dict(color='red', size=3))]
    for penetration_keypoint in penetration_keypoints:
        mesh = tm.primitives.Capsule(radius=0.01, height=0)
        v = mesh.vertices + penetration_keypoint
        f = mesh.faces
        penetration_keypoints_plotly += [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='burlywood', opacity=0.5)]
    penetration_labels_plotly = []
    if args.only_penetration_points:
        penetration_labels_plotly = [
            go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]],
                mode='text',
                text=[str(i + 1)],
                textfont=dict(size=10, color='black'),
                showlegend=False
            )
            for i, pt in enumerate(penetration_keypoints)
        ]

    collision_capsules_plotly = []
    if args.hand_model_type == 'dexhand021':
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
            if not args.only_boxes:
                collision_capsules_plotly.append(
                    go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='yellow', opacity=0.35)
                )
    else:
        # Shadow Hand collision proxies (capsules/boxes) in yellow
        for link_name in hand_model.mesh:
            geom_param = hand_model.mesh[link_name].get('geom_param')
            if geom_param is None:
                continue
            matrix = hand_model.current_status[link_name].get_matrix()[0]
            param = geom_param.detach().cpu().numpy()
            if param.shape[0] == 2:
                radius = float(param[0])
                height = float(param[1]) * 2.0
                p0 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device)
                p1 = torch.tensor([0.0, 0.0, height], dtype=torch.float, device=device)
                p0 = (p0 @ matrix[:3, :3].T) + matrix[:3, 3]
                p1 = (p1 @ matrix[:3, :3].T) + matrix[:3, 3]
                p0 = (p0 @ hand_model.global_rotation[0].T) + hand_model.global_translation[0]
                p1 = (p1 @ hand_model.global_rotation[0].T) + hand_model.global_translation[0]
                if not args.only_boxes:
                    cyl = tm.creation.cylinder(radius=radius, segment=[p0.detach().cpu().numpy(), p1.detach().cpu().numpy()])
                    v = cyl.vertices
                    f = cyl.faces
                    collision_capsules_plotly.append(
                        go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='yellow', opacity=0.35)
                    )
            elif param.shape[0] == 3:
                extents = 2.0 * param
                box = tm.creation.box(extents=extents)
                v = torch.tensor(box.vertices, dtype=torch.float, device=device)
                v = (v @ matrix[:3, :3].T) + matrix[:3, 3]
                v = (v @ hand_model.global_rotation[0].T) + hand_model.global_translation[0]
                v = v.detach().cpu().numpy()
                f = box.faces
                collision_capsules_plotly.append(
                    go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color='orange', opacity=0.6)
                )

    if args.only_boxes:
        fig = go.Figure(collision_capsules_plotly)
    elif args.only_penetration_points:
        fig = go.Figure(penetration_keypoints_plotly + penetration_labels_plotly)
    else:
        fig = go.Figure(hand_plotly + contact_candidates_plotly + contact_labels_plotly + penetration_keypoints_plotly + penetration_labels_plotly + collision_capsules_plotly)
    fig.update_layout(scene_aspectmode='data')
    fig.show()
