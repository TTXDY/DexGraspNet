"""
Simple ShadowHand visualization using HandModelMJCFLite (no pytorch3d required)
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.append(parent_dir)

import numpy as np
import torch
import transforms3d
from utils.hand_model_lite import HandModelMJCFLite


if __name__ == '__main__':
    device = torch.device('cpu')

    # Create hand model
    hand_model = HandModelMJCFLite(
        mjcf_path='mjcf/shadow_hand_wrist_free.xml',
        mesh_path='mjcf/meshes',
        device=device
    )

    # Set a sample hand pose (slightly curved fingers)
    joint_angles = torch.tensor([
        0.1, 0, 0.6, 0,      # First finger
        0, 0, 0.6, 0,        # Middle finger
        -0.1, 0, 0.6, 0,     # Ring finger
        0, -0.2, 0, 0.6, 0,  # Little finger
        0, 1.2, 0, -0.2, 0   # Thumb
    ], dtype=torch.float, device=device)

    # Set rotation and translation
    rotation = torch.tensor(
        transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'),
        dtype=torch.float, device=device
    )
    hand_pose = torch.cat([
        torch.tensor([0, 0, 0], dtype=torch.float, device=device),  # translation
        rotation.T.ravel()[:6],  # rotation (6D representation)
        joint_angles
    ])

    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # Get mesh and visualize
    hand_mesh = hand_model.get_trimesh_data(0)

    print(f"ShadowHand visualization")
    print(f"Number of vertices: {len(hand_mesh.vertices)}")
    print(f"Number of faces: {len(hand_mesh.faces)}")

    # Export to file (pyglet requires Python 3.8+)
    output_file = "shadowhand_visualization.obj"
    hand_mesh.export(output_file)
    print(f"\nMesh exported to: {output_file}")
    print("You can view this file with:")
    print("  - MeshLab")
    print("  - Blender")
    print("  - Online viewers like https://3dviewer.net/")
