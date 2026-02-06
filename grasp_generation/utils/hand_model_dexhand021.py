"""
HandModel adapter for dexhand021 hand
Adapted from hand_model.py for Shadow Hand

Key differences from Shadow Hand:
- 12-DOF control space (mapped to 20 joints) vs 22 DOF
- X-axis coordinate system (vs Z-axis)
- Explicit 6 DOF floating base in MJCF
- Uses fromto capsules (vs pos+size capsules)
- Link names: r_f_link{finger}_{segment} (vs robot0: prefix)
"""

import os
import json
import numpy as np
import torch
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import pytorch_kinematics as pk
import plotly.graph_objects as go
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
from torchsdf import index_vertices_by_faces
import xml.etree.ElementTree as ET
import transforms3d


def get_dexhand021_output_alignment(device=None, dtype=torch.float):
    rotation = transforms3d.euler.euler2mat(np.pi, np.deg2rad(-30), np.pi / 2, axes='sxyz')
    return torch.tensor(rotation, dtype=dtype, device=device)


class HandModelDexHand021:
    # Control-space names (12 inputs) and full joint names (20 joints)
    CONTROL_NAMES = [
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

    JOINT_NAMES = [
        "r_f_joint1_1", "r_f_joint1_2", "r_f_joint1_3", "r_f_joint1_4",
        "r_f_joint2_1", "r_f_joint2_2", "r_f_joint2_3", "r_f_joint2_4",
        "r_f_joint3_1", "r_f_joint3_2", "r_f_joint3_3", "r_f_joint3_4",
        "r_f_joint4_1", "r_f_joint4_2", "r_f_joint4_3", "r_f_joint4_4",
        "r_f_joint5_1", "r_f_joint5_2", "r_f_joint5_3", "r_f_joint5_4",
    ]
    @staticmethod
    def _parse_points_list(points, key, name):
        if points is None:
            return None
        if not isinstance(points, list):
            raise ValueError(f"{name} for {key} must be a list, got {type(points)}")
        if len(points) == 0:
            return torch.tensor([], dtype=torch.float32).reshape(0, 3)
        is_flat = isinstance(points[0], (float, int))
        is_nested = isinstance(points[0], list)
        if is_flat:
            if len(points) % 3 != 0:
                raise ValueError(f"{name} for {key} flat list length must be multiple of 3, got {len(points)}")
            return torch.tensor(points, dtype=torch.float32).reshape(-1, 3)
        if is_nested:
            for i, row in enumerate(points):
                if not isinstance(row, list) or len(row) != 3:
                    raise ValueError(f"{name} for {key} must be list of [x,y,z], bad row {i}: {row}")
            return torch.tensor(points, dtype=torch.float32)
        raise ValueError(f"{name} for {key} must be flat list or list of lists, got {type(points[0])}")

    @staticmethod
    def _load_capsules_from_mjcf(mjcf_path, device):
        capsules = {}
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        if worldbody is None:
            return capsules

        def recurse(body):
            body_name = body.get('name')
            for geom in body.findall('geom'):
                if geom.get('type') != 'capsule':
                    continue
                fromto = geom.get('fromto')
                if not fromto:
                    continue
                size = geom.get('size', '')
                radius = float(size.split()[0]) if size else 0.0
                values = [float(x) for x in fromto.split()]
                if len(values) != 6:
                    continue
                entry = {
                    'fromto': torch.tensor(values, dtype=torch.float, device=device).reshape(2, 3),
                    'radius': radius,
                }
                capsules.setdefault(body_name, []).append(entry)
                # Also map to "_child" names used by pytorch_kinematics links.
                capsules.setdefault(f"{body_name}_child", []).append(entry)
            for child in body.findall('body'):
                recurse(child)

        for body in worldbody.findall('body'):
            recurse(body)
        return capsules

    def __init__(self, mjcf_path, mesh_path, contact_points_path, penetration_points_path, n_surface_points=0, device='cpu'):
        """
        Create a Hand Model for dexhand021 MJCF robot

        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        mesh_path: str
            path to mesh directory
        contact_points_path: str
            path to hand-selected contact candidates
        penetration_points_path: str
            path to hand-selected penetration keypoints
        n_surface_points: int
            number of points to sample from surface of hand, use fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        self.output_align = torch.eye(3, dtype=torch.float, device=device)
        self.output_align_inv = self.output_align

        # load articulation

        self.chain = pk.build_chain_from_mjcf(open(mjcf_path, 'rb').read()).to(dtype=torch.float, device=device)
        self.joint_names_all = self.chain.get_joint_parameter_names()
        self.joint_name_to_all_index = {name: i for i, name in enumerate(self.joint_names_all)}
        self.base_joint_names = {'ARTx', 'ARTy', 'ARTz', 'ARRx', 'ARRy', 'ARRz'}
        self.base_joint_indices = [i for i, name in enumerate(self.joint_names_all) if name in self.base_joint_names]
        self.hand_joint_indices = [i for i, name in enumerate(self.joint_names_all) if name not in self.base_joint_names]
        self.n_dofs_all = len(self.joint_names_all)
        self.n_dofs = len(self.hand_joint_indices)

        # load contact points and penetration points

        contact_points = json.load(open(contact_points_path, 'r')) if contact_points_path is not None else None
        penetration_points = json.load(open(penetration_points_path, 'r')) if penetration_points_path is not None else None

        # build mesh

        self.mesh = {}
        areas = {}
        capsules_by_link = self._load_capsules_from_mjcf(mjcf_path, device)
        self.collision_capsule_radius = 0.01  # 2cm diameter (used only for custom capsules)
        self._warned_insufficient_capsules = False
        self.use_mjcf_collision_capsules = False

        def build_mesh_recurse(body):
            if(len(body.link.visuals) > 0):
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "box":
                        link_mesh = tm.load_mesh(os.path.join(mesh_path, 'box.obj'), process=False)
                        link_mesh.vertices *= visual.geom_param.detach().cpu().numpy()
                    elif visual.geom_type == "sphere":
                        link_mesh = tm.primitives.Sphere(radius=visual.geom_param[0])
                    elif visual.geom_type == "capsule":
                        if visual.geom_param.numel() < 2:
                            # Skip collision-only capsules that only define radius (fromto capsules).
                            continue
                        link_mesh = tm.primitives.Capsule(radius=visual.geom_param[0], height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
                    elif visual.geom_type == "mesh":
                        # dexhand021 mesh files are named like "r_f_link1_1.STL"
                        mesh_name = visual.geom_param[0]
                        if ":" in mesh_name:
                            mesh_name = mesh_name.split(":")[1]
                        mesh_file = os.path.join(mesh_path, mesh_name + ".STL")
                        if not os.path.exists(mesh_file):
                            mesh_file = os.path.join(mesh_path, mesh_name + ".obj")
                        link_mesh = tm.load_mesh(mesh_file, process=False)
                        if visual.geom_param[1] is not None:
                            scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(self.device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                contact_candidates = None
                if contact_points is not None and link_name in contact_points:
                    contact_candidates = self._parse_points_list(contact_points[link_name], link_name, "contact_points").to(device)
                penetration_keypoints = None
                if penetration_points is not None and link_name in penetration_points:
                    penetration_keypoints = self._parse_points_list(penetration_points[link_name], link_name, "penetration_points").to(device)
                self.mesh[link_name] = {
                    'vertices': link_vertices,
                    'faces': link_faces,
                    'contact_candidates': contact_candidates,
                    'penetration_keypoints': penetration_keypoints,
                }
                if link_name in capsules_by_link:
                    self.mesh[link_name]['capsules'] = capsules_by_link[link_name]
                # Cache face vertices for mesh SDF
                link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                self.mesh[link_name]['face_verts'] = link_face_verts
                areas[link_name] = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
            for children in body.children:
                build_mesh_recurse(children)
        build_mesh_recurse(self.chain._root)

        # set joint limits

        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(body.joint.range[0])
                self.joints_upper.append(body.joint.range[1])
            for children in body.children:
                set_joint_range_recurse(children)
        set_joint_range_recurse(self.chain._root)
        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)
        if self.base_joint_indices:
            hand_joint_indices = torch.tensor(self.hand_joint_indices, dtype=torch.long, device=device)
            self.joints_names = [self.joints_names[i] for i in self.hand_joint_indices]
            self.joints_lower = self.joints_lower.index_select(0, hand_joint_indices)
            self.joints_upper = self.joints_upper.index_select(0, hand_joint_indices)

        # Cache full joint limits/names (20 DOF) before switching to control space.
        full_lower_map = dict(zip(self.joints_names, self.joints_lower))
        full_upper_map = dict(zip(self.joints_names, self.joints_upper))
        self.joints_names_full = list(self.JOINT_NAMES)
        self.joints_lower_full = torch.stack([full_lower_map[name] for name in self.joints_names_full]).to(device)
        self.joints_upper_full = torch.stack([full_upper_map[name] for name in self.joints_names_full]).to(device)

        # Switch to 12-DOF control space for optimization.
        self.joints_names = list(self.CONTROL_NAMES)
        self.joints_lower = torch.tensor([
            0.0,  # ctrl_thumb_spread
            0.0,  # ctrl_thumb_mcp
            0.0,  # ctrl_thumb_dip
            0.0,  # ctrl_finger_spread
            0.0,  # ctrl_index_mcp
            0.0,  # ctrl_index_dip
            0.0,  # ctrl_middle_mcp
            0.0,  # ctrl_middle_dip
            0.0,  # ctrl_ring_mcp
            0.0,  # ctrl_ring_dip
            0.0,  # ctrl_pinky_mcp
            0.0,  # ctrl_pinky_dip
        ], dtype=torch.float, device=device)
        self.joints_upper = torch.tensor([
            2.2,  # ctrl_thumb_spread
            1.3,  # ctrl_thumb_mcp
            1.3,  # ctrl_thumb_dip
            0.3,  # ctrl_finger_spread (maps to pinky *2)
            1.3,  # ctrl_index_mcp
            1.3,  # ctrl_index_dip
            1.3,  # ctrl_middle_mcp
            1.3,  # ctrl_middle_dip
            1.3,  # ctrl_ring_mcp
            1.3,  # ctrl_ring_dip
            1.3,  # ctrl_pinky_mcp
            1.3,  # ctrl_pinky_dip
        ], dtype=torch.float, device=device)
        self.n_dofs = len(self.joints_names)

        self.joint_name_to_full_index = {name: i for i, name in enumerate(self.JOINT_NAMES)}

        # sample surface points

        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(0), self.mesh[link_name]['faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]['surface_points'] = surface_points

        # indexing

        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))

        self.contact_candidates = []
        self.global_index_to_link_index = []
        for link_name in self.mesh:
            candidates = self.mesh[link_name]['contact_candidates']
            if candidates is None:
                continue
            self.contact_candidates.append(candidates)
            self.global_index_to_link_index.extend([self.link_name_to_link_index[link_name]] * len(candidates))
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(self.global_index_to_link_index, dtype=torch.long, device=device)
        self.n_contact_candidates = self.contact_candidates.shape[0]

        self.penetration_keypoints = []
        self.global_index_to_link_index_penetration = []
        for link_name in self.mesh:
            keypoints = self.mesh[link_name]['penetration_keypoints']
            if keypoints is None:
                continue
            self.penetration_keypoints.append(keypoints)
            self.global_index_to_link_index_penetration.extend([self.link_name_to_link_index[link_name]] * len(keypoints))
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(self.global_index_to_link_index_penetration, dtype=torch.long, device=device)
        self.n_keypoints = self.penetration_keypoints.shape[0]

        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

    def set_parameters(self, hand_pose, contact_point_indices=None):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(self.hand_pose[:, 3:9])
        control_angles = self.hand_pose[:, 9:]
        full_joint_angles = self._controls_to_full_joint_angles(control_angles)
        self.current_status = self.chain.forward_kinematics(full_joint_angles)
        self.joint_angles_full = self._controls_to_joint_angles_full(control_angles)
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device)
            for link_name in self.mesh:
                if self.mesh[link_name]['contact_candidates'] is None:
                    continue
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, n_contact, 4, 4)
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat([self.contact_points, torch.ones(batch_size, n_contact, 1, dtype=torch.float, device=self.device)], dim=2)
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[:, :, :3, 0]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)

    def _controls_to_full_joint_angles(self, control_angles):
        batch_size = control_angles.shape[0]
        full_joint_angles = torch.zeros(batch_size, self.n_dofs_all, dtype=control_angles.dtype, device=self.device)

        def set_joint(name, value):
            idx = self.joint_name_to_all_index.get(name)
            if idx is not None:
                full_joint_angles[:, idx] = value

        # Thumb
        set_joint("r_f_joint1_1", control_angles[:, 0])
        set_joint("r_f_joint1_2", control_angles[:, 1])
        set_joint("r_f_joint1_3", control_angles[:, 2])
        set_joint("r_f_joint1_4", control_angles[:, 2])

        # Finger spread (index/ring/pinky with pinky 2x)
        spread = control_angles[:, 3]
        set_joint("r_f_joint2_1", spread)
        set_joint("r_f_joint4_1", spread)
        set_joint("r_f_joint5_1", 2.0 * spread)

        # Index
        set_joint("r_f_joint2_2", control_angles[:, 4])
        set_joint("r_f_joint2_3", control_angles[:, 5])
        set_joint("r_f_joint2_4", control_angles[:, 5])

        # Middle (joint3_1 locked to 0)
        set_joint("r_f_joint3_1", torch.zeros_like(control_angles[:, 0]))
        set_joint("r_f_joint3_2", control_angles[:, 6])
        set_joint("r_f_joint3_3", control_angles[:, 7])
        set_joint("r_f_joint3_4", control_angles[:, 7])

        # Ring
        set_joint("r_f_joint4_2", control_angles[:, 8])
        set_joint("r_f_joint4_3", control_angles[:, 9])
        set_joint("r_f_joint4_4", control_angles[:, 9])

        # Pinky
        set_joint("r_f_joint5_2", control_angles[:, 10])
        set_joint("r_f_joint5_3", control_angles[:, 11])
        set_joint("r_f_joint5_4", control_angles[:, 11])

        return full_joint_angles

    def _controls_to_joint_angles_full(self, control_angles):
        batch_size = control_angles.shape[0]
        joint_angles = torch.zeros(batch_size, len(self.JOINT_NAMES), dtype=control_angles.dtype, device=self.device)

        def set_joint(name, value):
            idx = self.joint_name_to_full_index.get(name)
            if idx is not None:
                joint_angles[:, idx] = value

        # Thumb
        set_joint("r_f_joint1_1", control_angles[:, 0])
        set_joint("r_f_joint1_2", control_angles[:, 1])
        set_joint("r_f_joint1_3", control_angles[:, 2])
        set_joint("r_f_joint1_4", control_angles[:, 2])

        # Finger spread (index/ring/pinky with pinky 2x)
        spread = control_angles[:, 3]
        set_joint("r_f_joint2_1", spread)
        set_joint("r_f_joint4_1", spread)
        set_joint("r_f_joint5_1", 2.0 * spread)

        # Index
        set_joint("r_f_joint2_2", control_angles[:, 4])
        set_joint("r_f_joint2_3", control_angles[:, 5])
        set_joint("r_f_joint2_4", control_angles[:, 5])

        # Middle (joint3_1 locked to 0)
        set_joint("r_f_joint3_1", torch.zeros_like(control_angles[:, 0]))
        set_joint("r_f_joint3_2", control_angles[:, 6])
        set_joint("r_f_joint3_3", control_angles[:, 7])
        set_joint("r_f_joint3_4", control_angles[:, 7])

        # Ring
        set_joint("r_f_joint4_2", control_angles[:, 8])
        set_joint("r_f_joint4_3", control_angles[:, 9])
        set_joint("r_f_joint4_4", control_angles[:, 9])

        # Pinky
        set_joint("r_f_joint5_2", control_angles[:, 10])
        set_joint("r_f_joint5_3", control_angles[:, 11])
        set_joint("r_f_joint5_4", control_angles[:, 11])

        return joint_angles

    def controls_to_joint_angles(self, control_angles):
        if control_angles.dim() == 1:
            control_angles = control_angles.unsqueeze(0)
        if control_angles.device != self.device:
            control_angles = control_angles.to(self.device)
        return self._controls_to_joint_angles_full(control_angles)

    def recompute_fk(self):
        if self.hand_pose is None:
            return
        control_angles = self.hand_pose[:, 9:]
        full_joint_angles = self._controls_to_full_joint_angles(control_angles)
        self.current_status = self.chain.forward_kinematics(full_joint_angles)
    # NOTE: Single-sample helper for debugging/visualization; batch logic uses _link_origin_world_batch.
    def _link_origin_world(self, link_name):
        matrix = self.current_status[link_name].get_matrix()[0]
        origin = matrix[:3, 3]
        return origin @ self.global_rotation[0].T + self.global_translation[0]

    # NOTE: Single-sample helper for debugging/visualization; batch logic uses _link_keypoints_world_batch.
    def _link_keypoints_world(self, link_name):
        keypoints = self.mesh[link_name].get('penetration_keypoints')
        if keypoints is None or keypoints.numel() == 0:
            return []
        pts = self.current_status[link_name].transform_points(keypoints)
        if pts.dim() == 3:
            pts = pts[0]
        pts = pts @ self.global_rotation[0].T + self.global_translation[0]
        return [p for p in pts]

    def _link_origin_world_batch(self, link_name):
        matrix = self.current_status[link_name].get_matrix()
        batch_size = self.global_translation.shape[0]
        if matrix.shape[0] == 1 and batch_size > 1:
            matrix = matrix.expand(batch_size, 4, 4)
        origin = matrix[:, :3, 3]
        origin = torch.bmm(origin.unsqueeze(1), self.global_rotation.transpose(1, 2)).squeeze(1)
        return origin + self.global_translation

    def _link_keypoints_world_batch(self, link_name):
        keypoints = self.mesh[link_name].get('penetration_keypoints')
        if keypoints is None or keypoints.numel() == 0:
            return None
        pts = self.current_status[link_name].transform_points(keypoints)
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)
        pts = pts @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return pts

    def _build_palm_box_world_batch(self):
        # Build a palm box from link1 positions (index/middle/ring/pinky) and right_hand_base.
        required = [
            "r_f_link2_1_child",  # index
            "r_f_link3_1_child",  # middle
            "r_f_link4_1_child",  # ring
            "r_f_link5_1_child",  # pinky
            "right_hand_base",
        ]
        batch_size = self.global_translation.shape[0]
        for name in required:
            if name not in self.mesh:
                return None
        p_index = self._link_origin_world_batch("r_f_link2_1_child")
        p_middle = self._link_origin_world_batch("r_f_link3_1_child")
        p_ring = self._link_origin_world_batch("r_f_link4_1_child")
        p_pinky = self._link_origin_world_batch("r_f_link5_1_child")
        p_base = self._link_origin_world_batch("right_hand_base")

        axes = self.global_rotation
        x_axis = axes[:, :, 0]
        y_axis = axes[:, :, 1]
        z_axis = axes[:, :, 2]
        # Fixed palm box size (meters)
        width_len = torch.full((batch_size,), 0.15, dtype=self.global_translation.dtype, device=self.device)
        length_len = torch.full((batch_size,), 0.10, dtype=self.global_translation.dtype, device=self.device)
        height = 0.02
        extents = torch.stack([width_len * 0.5, length_len * 0.5, torch.full_like(width_len, height * 0.5)], dim=1)
        center = p_base + 0.5 * width_len.unsqueeze(1) * x_axis
        return center, axes, extents

    def _build_collision_capsules_world(self):
        batch_size = self.global_translation.shape[0]
        if self.use_mjcf_collision_capsules:
            capsules = []
            for link_name in self.mesh:
                link_caps = self.mesh[link_name].get('capsules')
                if not link_caps:
                    continue
                matrix = self.current_status[link_name].get_matrix()
                for capsule in link_caps:
                    p0 = capsule['fromto'][0].to(self.device)
                    p1 = capsule['fromto'][1].to(self.device)
                    radius = float(capsule['radius'])
                    p0 = p0.unsqueeze(0).expand(batch_size, 3)
                    p1 = p1.unsqueeze(0).expand(batch_size, 3)
                    p0 = torch.bmm(p0.unsqueeze(1), matrix[:, :3, :3].transpose(1, 2)).squeeze(1) + matrix[:, :3, 3]
                    p1 = torch.bmm(p1.unsqueeze(1), matrix[:, :3, :3].transpose(1, 2)).squeeze(1) + matrix[:, :3, 3]
                    p0 = torch.bmm(p0.unsqueeze(1), self.global_rotation.transpose(1, 2)).squeeze(1) + self.global_translation
                    p1 = torch.bmm(p1.unsqueeze(1), self.global_rotation.transpose(1, 2)).squeeze(1) + self.global_translation
                    capsules.append((p0, p1, radius))
            return capsules
        capsules = []
        radius = self.collision_capsule_radius
        for finger in [1, 2, 3, 4, 5]:
            base_link = f"r_f_link{finger}_1_child"
            base = self._link_origin_world_batch(base_link)
            keypoints2 = self._link_keypoints_world_batch(f"r_f_link{finger}_2_child")
            keypoints3 = self._link_keypoints_world_batch(f"r_f_link{finger}_3_child")
            keypoints4 = self._link_keypoints_world_batch(f"r_f_link{finger}_4_child")
            if finger == 1:
                if keypoints2 is not None and keypoints3 is not None and keypoints4 is not None and \
                   keypoints2.shape[1] >= 1 and keypoints3.shape[1] >= 1 and keypoints4.shape[1] >= 1:
                    p1, p2, p3 = keypoints2[:, 0], keypoints3[:, 0], keypoints4[:, 0]
                    capsules.extend([(base, p1, radius), (p1, p2, radius), (p2, p3, radius)])
                elif keypoints2 is not None and keypoints3 is not None and \
                        keypoints2.shape[1] >= 1 and keypoints3.shape[1] >= 1:
                    p1, p2 = keypoints2[:, 0], keypoints3[:, 0]
                    p3 = p2 + (p2 - p1)
                    capsules.extend([(base, p1, radius), (p1, p2, radius), (p2, p3, radius)])
                elif keypoints2 is not None and keypoints2.shape[1] >= 1:
                    capsules.append((base, keypoints2[:, 0], radius))
            else:
                if keypoints2 is not None and keypoints3 is not None and keypoints4 is not None and \
                   keypoints2.shape[1] >= 1 and keypoints3.shape[1] >= 1 and keypoints4.shape[1] >= 2:
                    p2 = keypoints2[:, 0]
                    p3 = keypoints3[:, 0]
                    # use the second point on link4 as the "extension" point
                    p4 = keypoints4[:, 1]
                    capsules.extend([(base, p2, radius), (p2, p3, radius), (p3, p4, radius)])
                elif keypoints2 is not None and keypoints3 is not None and \
                        keypoints2.shape[1] >= 1 and keypoints3.shape[1] >= 1:
                    p2, p3 = keypoints2[:, 0], keypoints3[:, 0]
                    p4 = p3 + (p3 - p2)
                    capsules.extend([(base, p2, radius), (p2, p3, radius), (p3, p4, radius)])
                elif keypoints2 is not None and keypoints2.shape[1] >= 1:
                    capsules.append((base, keypoints2[:, 0], radius))
                    if not self._warned_insufficient_capsules:
                        print("[DexHand021] Warning: insufficient penetration keypoints for 3 capsules per finger.")
                        self._warned_insufficient_capsules = True
        return capsules

    def cal_distance(self, x):
        """
        Calculate signed distances from object point clouds to hand surface meshes

        Interiors are positive, exteriors are negative

        Use analytical method for capsules and modified Kaolin package for meshes

        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """
        # Consider each link separately:
        #   First, transform x into each link's local reference frame using inversed fk, which gives us x_local
        #   Next, calculate point-to-primitive distances in each link's frame, this gives dis_local
        #   Finally, the maximum over all links is the final distance from one point to the entire articulation
        dis = []
        if x.dim() != 3:
            raise ValueError(f"x must be (B, N, 3), got {x.shape}")
        batch_size = self.global_translation.shape[0]
        if x.shape[0] != batch_size:
            if x.shape[1] == batch_size:
                x = x.transpose(0, 1)
            else:
                raise ValueError(f"x batch size {x.shape[0]} does not match {batch_size} and cannot be inferred")
        if x.shape[-1] != 3:
            if x.shape[1] == 3:
                x = x.transpose(1, 2)
            else:
                raise ValueError(f"x must be (B, N, 3), got {x.shape}")
        x_world = x
        capsules = self._build_collision_capsules_world()
        palm_box = self._build_palm_box_world_batch()
        for p0, p1, radius in capsules:
            if p0.dim() == 1:
                p0 = p0.unsqueeze(0)
                p1 = p1.unsqueeze(0)
            if p0.shape[-1] != 3 or p1.shape[-1] != 3:
                raise ValueError(f"Capsule endpoints must be (..., 3), got {p0.shape} and {p1.shape}")
            if p0.shape[0] != x_world.shape[0]:
                raise ValueError(f"Capsule batch size {p0.shape[0]} does not match x batch {x_world.shape[0]}")
            v = p1 - p0
            vv = (v * v).sum(dim=1).clamp_min(1e-12)
            w = x_world - p0.unsqueeze(1)
            t = (w * v.unsqueeze(1)).sum(dim=2) / vv.unsqueeze(1)
            t = t.clamp(0.0, 1.0)
            closest = p0.unsqueeze(1) + t.unsqueeze(2) * v.unsqueeze(1)
            d = radius - (x_world - closest).norm(dim=2)
            dis.append(d)
        if palm_box is not None:
            center, axes, extents = palm_box
            x_local = torch.bmm(x_world - center.unsqueeze(1), axes)
            q = x_local.abs() - extents.unsqueeze(1)
            outside = torch.clamp(q, min=0.0)
            outside_dist = outside.norm(dim=2)
            inside = torch.minimum(torch.maximum(q[..., 0], torch.maximum(q[..., 1], q[..., 2])), torch.zeros_like(q[..., 0]))
            sdf = outside_dist + inside
            dis.append(-sdf)
        if not dis:
            raise RuntimeError("No collision capsules found for DexHand021 distance computation.")
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(self):
        """
        Calculate self penetration energy

        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        batch_size = self.global_translation.shape[0]
        points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = self.global_index_to_link_index_penetration.clone().repeat(batch_size, 1)
        transforms = torch.zeros(batch_size, self.n_keypoints, 4, 4, dtype=torch.float, device=self.device)
        for link_name in self.mesh:
            if self.mesh[link_name]['penetration_keypoints'] is None:
                continue
            mask = link_indices == self.link_name_to_link_index[link_name]
            cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, self.n_keypoints, 4, 4)
            transforms[mask] = cur[mask]
        points = torch.cat([points, torch.ones(batch_size, self.n_keypoints, 1, dtype=torch.float, device=self.device)], dim=2)
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(dis < 1e-6, 1e6 * torch.ones_like(dis), dis)
        dis = 0.02 - dis
        E_spen = torch.where(dis > 0, dis, torch.zeros_like(dis))
        return E_spen.sum((1, 2))

    def get_surface_points(self):
        """
        Get surface points

        Returns
        -------
        points: (N, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['surface_points'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['surface_points']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        # no output alignment
        return points

    def get_mesh_vertices(self):
        """
        Get all mesh vertices in world coordinates.

        Returns
        -------
        vertices: (B, V, 3) torch.Tensor
            concatenated mesh vertices
        """
        vertices = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]['vertices'])
            if v.dim() == 2:
                v = v.unsqueeze(0)
            if v.shape[0] == 1 and batch_size > 1:
                v = v.expand(batch_size, -1, -1)
            vertices.append(v)
        vertices = torch.cat(vertices, dim=1).to(self.device)
        vertices = vertices @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        # no output alignment
        return vertices

    def ground_penetration(self, ground_height):
        """
        Compute hand-ground penetration using collision capsules and palm box.

        Parameters
        ----------
        ground_height: (B,) torch.Tensor
            ground plane height per batch

        Returns
        -------
        penetration: (B,) torch.Tensor
            summed penetration depths below ground
        """
        if ground_height.dim() != 1:
            ground_height = ground_height.view(-1)
        batch_size = self.global_translation.shape[0]
        if ground_height.shape[0] != batch_size:
            raise ValueError(f"ground_height batch {ground_height.shape[0]} != {batch_size}")
        penetration = torch.zeros(batch_size, dtype=self.global_translation.dtype, device=self.device)

        capsules = self._build_collision_capsules_world()
        for p0, p1, radius in capsules:
            min_z = torch.minimum(p0[:, 2], p1[:, 2]) - radius
            penetration += (ground_height - min_z).clamp(min=0.0)

        palm_box = self._build_palm_box_world_batch()
        if palm_box is not None:
            center, axes, extents = palm_box  # axes: (B,3,3)
            # min z = center.z - sum_i |R_{z,i}| * extent_i
            z_span = (axes[:, 2, :].abs() * extents).sum(dim=1)
            min_z = center[:, 2] - z_span
            penetration += (ground_height - min_z).clamp(min=0.0)

        return penetration

    def get_contact_candidates(self):
        """
        Get all contact candidates

        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            if self.mesh[link_name]['contact_candidates'] is None:
                continue
            n_surface_points = self.mesh[link_name]['contact_candidates'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['contact_candidates']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        # no output alignment
        return points

    def get_penetraion_keypoints(self):
        """
        Get penetration keypoints

        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            if self.mesh[link_name]['penetration_keypoints'] is None:
                continue
            n_surface_points = self.mesh[link_name]['penetration_keypoints'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['penetration_keypoints']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        # no output alignment
        return points

    def get_plotly_data(self, i, opacity=0.5, color='lightblue', with_contact_points=False, pose=None):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            # no output alignment
            data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=color, opacity=opacity))
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            # no output alignment
            data.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2], mode='markers', marker=dict(color='red', size=5)))
        return data
