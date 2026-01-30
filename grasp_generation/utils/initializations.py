"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: initializations
Modified: 2026.01.26 - Added support for dexhand021
"""

import torch
import transforms3d
import math
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
import numpy as np
import torch.nn.functional

# Import hand-specific configurations
try:
    from config.dexhand021_default_angles import joint_angles_mu_dexhand021
except ImportError:
    joint_angles_mu_dexhand021 = None


def initialize_convex_hull(hand_model, object_model, args):
    """
    Initialize grasp translation, rotation, joint angles, and contact point indices
    
    Parameters
    ----------
    hand_model: hand_model.HandModel
    object_model: object_model.ObjectModel
    args: Namespace
    """

    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each

    # Detect hand model type and set hand-specific parameters
    hand_model_type = type(hand_model).__name__

    if hand_model_type == 'HandModelDexHand021':
        # dexhand021: 12-DOF control space. Finger forward is +X in the native frame.
        if joint_angles_mu_dexhand021 is None:
            raise ImportError("dexhand021 default angles not found. Please check config/dexhand021_default_angles.py")
        joint_angles_mu = joint_angles_mu_dexhand021.to(device=device, dtype=torch.float)
        # Predefined initialization orientations (randomly chosen per sample)
        rotation_hand_candidates = [
            torch.tensor(transforms3d.euler.euler2mat(np.pi / 2, 0, 0, axes='sxyz'), dtype=torch.float, device=device),
            torch.tensor(transforms3d.euler.euler2mat(0, 0, 0, axes='sxyz'), dtype=torch.float, device=device),
            torch.tensor(transforms3d.euler.euler2mat(np.pi / 2, 0, np.pi / 2, axes='sxyz'), dtype=torch.float, device=device),
            torch.tensor(transforms3d.euler.euler2mat(0, 0, -np.pi / 2, axes='sxyz'), dtype=torch.float, device=device),
        ]
        approach_direction = torch.tensor([0, 0, 1], dtype=torch.float, device=device)
        # Adjust sampling ranges for the larger dexhand021 geometry
        distance_lower = args.distance_lower * args.dexhand_distance_scale
        distance_upper = args.distance_upper * args.dexhand_distance_scale
        theta_lower = args.theta_lower * args.dexhand_theta_scale
        theta_upper = args.theta_upper * args.dexhand_theta_scale
    else:
        # Shadow Hand: 22 DOF, Z-axis coordinate system
        joint_angles_mu = torch.tensor([0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0, 0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0], dtype=torch.float, device=device)
        rotation_hand = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device)
        approach_direction = torch.tensor([0, 0, 1], dtype=torch.float, device=device)  # Z-axis for Shadow Hand
        distance_lower = args.distance_lower
        distance_upper = args.distance_upper
        theta_lower = args.theta_lower
        theta_upper = args.theta_upper

    # initialize translation and rotation

    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    for i in range(n_objects):
        
        # get inflated convex hull

        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        if hasattr(mesh_origin, 'nondegenerate_faces'):
            mesh_origin.update_faces(mesh_origin.nondegenerate_faces())
        obj_bounds = mesh_origin.bounds
        obj_height = float(obj_bounds[1][2] - obj_bounds[0][2])
        vertices += 0.2 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # sample points

        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * batch_size_each)
        p = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=batch_size_each)[0][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        # sample parameters

        distance = distance_lower + (distance_upper - distance_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        deviate_theta = theta_lower + (theta_upper - theta_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        process_theta = torch.zeros([batch_size_each], dtype=torch.float, device=device)
        rotate_theta = torch.zeros([batch_size_each], dtype=torch.float, device=device)

        # solve transformation
        # rotation_hand: rotate the hand to align its grasping direction with the +z axis
        # rotation_local: jitter the hand's orientation in a cone
        # rotation_global and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull

        rotation_local = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        rotation_global = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_size_each):
            rotation_local[j] = torch.eye(3, dtype=torch.float, device=device)
            rotation_global[j] = torch.eye(3, dtype=torch.float, device=device)

        # Use hand-specific approach direction (defined above based on hand model type)
        translation[i * batch_size_each: (i + 1) * batch_size_each] = p - distance.unsqueeze(1) * (rotation_global @ rotation_local @ approach_direction.reshape(1, -1, 1)).squeeze(2)
        # Use hand-specific rotation_hand (defined above based on hand model type)
        if hand_model_type == 'HandModelDexHand021':
            choices = torch.randint(0, len(rotation_hand_candidates), (batch_size_each,), device=device)
            rotation_hand = torch.stack([rotation_hand_candidates[c] for c in choices], dim=0)
            rotation[i * batch_size_each: (i + 1) * batch_size_each] = rotation_hand
            # Apply base position constraints per rotation choice.
            sl = slice(i * batch_size_each, (i + 1) * batch_size_each)
            t = translation[sl]
            eps = 1e-6
            x = t[:, 0]
            y = t[:, 1]
            z = t[:, 2]
            # All choices: base x > 0
            x = torch.where(x <= 0, -x + eps, x)
            # Choice 1 (index 1): base z fixed to 0.3, base x > 0
            mask1 = choices == 1
            z = torch.where(mask1, torch.full_like(z, 0.3), z)
            # Other choices: base z > 0
            mask_other = ~mask1
            z = torch.where(mask_other & (z <= 0), -z + eps, z)
            # Choice 2 (index 2): base x > 0, base y < 0
            mask2 = choices == 2
            y = torch.where(mask2 & (y >= 0), -y - 0.01, y)
            # Choice 3 (index 3): base z > 0, base x/y near 0 with ~2cm jitter
            mask3 = choices == 3
            jitter = 0.02 * (2.0 * torch.rand_like(x) - 1.0)
            x = torch.where(mask3, jitter, x)
            y = torch.where(mask3, jitter, y)
            t[:, 0] = x
            t[:, 1] = y
            t[:, 2] = z
            translation[sl] = t
        else:
            rotation[i * batch_size_each: (i + 1) * batch_size_each] = rotation_global @ rotation_local @ rotation_hand
    
    # initialize joint angles
    # joint_angles_mu and rotation_hand are already defined above based on hand model type
    # use truncated normal distribution to jitter the joint angles

    joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i], hand_model.joints_lower[i] - 1e-6, hand_model.joints_upper[i] + 1e-6)

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose.requires_grad_()

    # initialize contact point indices

    contact_point_indices = torch.randint(hand_model.n_contact_candidates, size=[total_batch_size, args.n_contact], device=device)

    hand_model.set_parameters(hand_pose, contact_point_indices)
