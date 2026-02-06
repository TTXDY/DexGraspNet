"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: generate grasps in large-scale, use multiple graphics cards, no logging
"""

import os
import sys

sys.path.append(os.path.realpath('.'))

import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import math
import random
import transforms3d
import json

from utils.hand_model import HandModel
from utils.hand_model_dexhand021 import HandModelDexHand021, get_dexhand021_output_alignment
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')

def _parse_contact_links(text):
    if text is None:
        return None
    s = text.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    s = s.replace('"', '').replace("'", '')
    tokens = [t.strip() for t in s.split(',') if t.strip()]
    return tokens or None


def _load_contact_links_map(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _resolve_contact_links(text, link_map):
    if text is None:
        return None, None
    key = text.strip()
    if key.lower() == "all":
        return None, "all"
    if key in link_map:
        return link_map[key], key
    tokens = _parse_contact_links(text)
    if tokens and all(t in link_map for t in tokens):
        expanded = []
        for t in tokens:
            expanded.extend(link_map[t])
        return expanded, ",".join(tokens)
    return tokens, "custom"


def _expand_contact_link_runs(text, link_map):
    if text is None:
        return [(None, None)]
    key = text.strip()
    if key.lower() == "all":
        if link_map:
            return [(link_map[k], k) for k in link_map.keys()]
        return [(None, "all")]
    if key in link_map:
        return [(link_map[key], key)]
    tokens = _parse_contact_links(text)
    if tokens and all(t in link_map for t in tokens):
        if len(tokens) == 1:
            t = tokens[0]
            return [(link_map[t], t)]
        return [(link_map[t], t) for t in tokens]
    return [(tokens, "custom")]


def _map_contact_token(token, hand_model_type):
    import re
    m = re.match(r'^(thumb|index|middle|ring|pinky)(\d+)$', token)
    if not m:
        raise ValueError(f"Invalid contact token '{token}'. Use like index4, thumb4.")
    finger = m.group(1)
    link_num = int(m.group(2))
    if hand_model_type == 'dexhand021':
        if link_num < 1 or link_num > 3:
            raise ValueError(f"dexhand021 link number must be 1-3 (joint1 excluded), got {link_num} in '{token}'.")
        finger_map = {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}
        return f"r_f_link{finger_map[finger]}_{link_num + 1}_child"
    else:
        if link_num < 1 or link_num > 3:
            raise ValueError(f"shadow_hand link number must be 1-3, got {link_num} in '{token}'.")
        finger_map = {'index': 'ff', 'middle': 'mf', 'ring': 'rf', 'pinky': 'lf', 'thumb': 'th'}
        seg_map = {1: 'proximal_child', 2: 'middle_child', 3: 'distal_child'}
        return f"robot0:{finger_map[finger]}{seg_map[link_num]}"


def generate(args_list):
    args, object_code_list, id, gpu_list = args_list

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare models

    n_objects = len(object_code_list)

    worker = multiprocessing.current_process()._identity[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[worker - 1]
    device = torch.device('cuda')

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

    object_model = ObjectModel(
        data_root_path=args.data_root_path,
        batch_size_each=args.batch_size_each,
        num_surface_samples=args.object_num_samples,
        device=device
    )
    object_model.initialize(object_code_list)
    if not args.random_obj_scale:
        object_model.object_scale_tensor = torch.ones_like(object_model.object_scale_tensor)

    contact_links_map = _load_contact_links_map(args.contact_links_file)
    if args.hand_model_type == 'dexhand021' and args.random_hand:
        contact_runs = [(None, None)]
    else:
        contact_runs = _expand_contact_link_runs(args.contact_links, contact_links_map)

    data_lists_by_object = {object_code: [] for object_code in object_code_list}
    skipped_e_pen = 0
    skipped_overflow = 0

    use_ground = args.hand_model_type == 'dexhand021' and not args.random_hand and args.w_ground > 0
    ground_height = None
    if use_ground:
        heights = []
        for i, mesh in enumerate(object_model.object_mesh_list):
            z_min = float(mesh.vertices[:, 2].min())
            heights.append(object_model.object_scale_tensor[i] * z_min)
        ground_height = torch.cat(heights, dim=0).to(device)

    for contact_tokens, contact_links_id in contact_runs:
        if contact_tokens is not None:
            args.n_contact = len(contact_tokens)

        initialize_convex_hull(hand_model, object_model, args)

        if contact_tokens is not None:
            link_indices = []
            total_batch_size = len(object_code_list) * args.batch_size_each
            for token in contact_tokens:
                link_name = _map_contact_token(token, args.hand_model_type)
                if link_name not in hand_model.link_name_to_link_index:
                    raise ValueError(f"Link '{link_name}' not found in hand model.")
                link_idx = hand_model.link_name_to_link_index[link_name]
                candidate_indices = (hand_model.global_index_to_link_index == link_idx).nonzero(as_tuple=False).squeeze(1)
                if candidate_indices.numel() == 0:
                    raise ValueError(f"No contact candidates found for link '{link_name}'.")
                link_indices.append(candidate_indices)
            idx_per_token = []
            for candidate_indices in link_indices:
                rand_idx = torch.randint(0, candidate_indices.numel(), (total_batch_size,), device=device)
                idx_per_token.append(candidate_indices[rand_idx])
            contact_point_indices = torch.stack(idx_per_token, dim=1)
            hand_model.set_parameters(hand_model.hand_pose, contact_point_indices)
            hand_model.allowed_contact_indices_per_slot = link_indices
        else:
            contact_links_id = contact_links_id or None
            hand_model.allowed_contact_indices_per_slot = None
        
        hand_pose_st = hand_model.hand_pose.detach()

        optim_config = {
            'switch_possibility': args.switch_possibility,
            'starting_temperature': args.starting_temperature,
            'temperature_decay': args.temperature_decay,
            'annealing_period': args.annealing_period,
            'step_size': args.step_size,
            'stepsize_period': args.stepsize_period,
            'mu': args.mu,
            'device': device
        }
        optimizer = Annealing(hand_model, **optim_config)

        # optimize
        
        weight_dict = dict(
            w_dis=args.w_dis,
            w_pen=args.w_pen,
            w_spen=args.w_spen,
            w_joints=args.w_joints,
            w_ground=args.w_ground if use_ground else 0.0,
        )
        energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_ground = cal_energy(
            hand_model, object_model, ground_height=ground_height, verbose=True, **weight_dict
        )

        energy.sum().backward(retain_graph=True)

        for step in tqdm(range(1, args.n_iter + 1), desc=f'optimizing (worker {worker})'):
            optimizer.try_step()

            optimizer.zero_grad()
            new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints, new_E_ground = cal_energy(
                hand_model, object_model, ground_height=ground_height, verbose=True, **weight_dict
            )

            new_energy.sum().backward(retain_graph=True)

            with torch.no_grad():
                accept, t = optimizer.accept_step(energy, new_energy)

                energy[accept] = new_energy[accept]
                E_dis[accept] = new_E_dis[accept]
                E_fc[accept] = new_E_fc[accept]
                E_pen[accept] = new_E_pen[accept]
                E_spen[accept] = new_E_spen[accept]
                E_joints[accept] = new_E_joints[accept]
                E_ground[accept] = new_E_ground[accept]


        # save results
        translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
        rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
        if args.hand_model_type == 'dexhand021':
            joint_names_full = hand_model.joints_names_full
            control_names = hand_model.joints_names
        else:
            joint_names = [
                'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
                'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
                'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
                'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
                'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
            ]
        full_contact_indices = hand_model.contact_point_indices.detach() if hand_model.contact_point_indices is not None else None
        for i, object_code in enumerate(object_code_list):
            for j in range(args.batch_size_each):
                idx = i * args.batch_size_each + j
                if len(data_lists_by_object[object_code]) >= args.target_batch_size_each:
                    skipped_overflow += 1
                    continue
                scale = object_model.object_scale_tensor[i][j].item()
                contact_point_indices = full_contact_indices[idx].detach().cpu().tolist() if full_contact_indices is not None else None
                hand_pose = hand_model.hand_pose[idx].detach().cpu()
                if args.hand_model_type == 'dexhand021':
                    joint_angles_full = hand_model.controls_to_joint_angles(hand_pose[9:]).squeeze(0).detach().cpu()
                    qpos = dict(zip(joint_names_full, joint_angles_full.tolist()))
                    controls = dict(zip(control_names, hand_pose[9:].tolist()))
                else:
                    qpos = dict(zip(joint_names, hand_pose[9:].tolist()))
                    controls = None
                rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
                if args.hand_model_type == 'dexhand021':
                    align = get_dexhand021_output_alignment(device=hand_pose.device)
                    rot = align @ rot
                    translation = align @ hand_pose[:3]
                else:
                    translation = hand_pose[:3]
                euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
                qpos.update(dict(zip(rot_names, euler)))
                qpos.update(dict(zip(translation_names, translation.tolist())))
                hand_pose_3_3_12 = None
                intrinsic_euler_hand_pose_3_3_12 = None
                if controls is not None:
                    hand_pose_3_3_12 = translation.tolist() + list(euler) + [controls[name] for name in control_names]
                    euler_intr = transforms3d.euler.mat2euler(rot, axes='rxyz')
                    intrinsic_euler_hand_pose_3_3_12 = translation.tolist() + list(euler_intr) + [controls[name] for name in control_names]
                hand_pose = hand_pose_st[idx].detach().cpu()
                if args.hand_model_type == 'dexhand021':
                    joint_angles_full_st = hand_model.controls_to_joint_angles(hand_pose[9:]).squeeze(0).detach().cpu()
                    qpos_st = dict(zip(joint_names_full, joint_angles_full_st.tolist()))
                    controls_st = dict(zip(control_names, hand_pose[9:].tolist()))
                else:
                    qpos_st = dict(zip(joint_names, hand_pose[9:].tolist()))
                    controls_st = None
                rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose[3:9].unsqueeze(0))[0]
                if args.hand_model_type == 'dexhand021':
                    align = get_dexhand021_output_alignment(device=hand_pose.device)
                    rot = align @ rot
                    translation = align @ hand_pose[:3]
                else:
                    translation = hand_pose[:3]
                euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
                qpos_st.update(dict(zip(rot_names, euler)))
                qpos_st.update(dict(zip(translation_names, translation.tolist())))
                e_pen_val = E_pen[idx].item()
                if args.max_e_pen is not None and e_pen_val > args.max_e_pen:
                    skipped_e_pen += 1
                    continue
                data_lists_by_object[object_code].append(dict(
                    scale=scale,
                    qpos=qpos,
                    qpos_st=qpos_st,
                    hand_pose_3_3_12=hand_pose_3_3_12,
                    intrinsic_euler_hand_pose_3_3_12=intrinsic_euler_hand_pose_3_3_12,
                    controls=controls,
                    controls_st=controls_st,
                    contact_point_indices=contact_point_indices,
                    contact_links_id=contact_links_id,
                    contact_links_tokens=contact_tokens,
                    energy=energy[idx].item(),
                    E_fc=E_fc[idx].item(),
                    E_dis=E_dis[idx].item(),
                    E_pen=e_pen_val,
                    E_spen=E_spen[idx].item(),
                    E_joints=E_joints[idx].item(),
                    E_ground=E_ground[idx].item(),
                ))
    for object_code, data_list in data_lists_by_object.items():
        np.save(os.path.join(args.result_path, object_code + '.npy'), data_list, allow_pickle=True)
    if args.max_e_pen is not None:
        print(f"Filtered out {skipped_e_pen} grasps with E_pen > {args.max_e_pen}")
    print(f"Skipped {skipped_overflow} grasps after reaching target batch size per object")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--result_path', default="../data/graspdata", type=str)
    parser.add_argument('--name', default=None, type=str,
                        help='Experiment name. If set and --result_path not provided, save to ../data/experiments/<name>/results')
    parser.add_argument('--data_root_path', default="../data/meshdata", type=str)
    parser.add_argument('--hand_model_type', default='shadow_hand', type=str, choices=['shadow_hand', 'dexhand021'])
    parser.add_argument('--object_code_list', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--todo', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_contact', default=4, type=int)
    parser.add_argument('--batch_size_each', default=500, type=int)
    parser.add_argument('--max_total_batch_size', default=1000, type=int)
    parser.add_argument('--n_iter', default=6000, type=int)
    parser.add_argument('--contact_links', default=None, type=str,
                        help='Override contact selection with link tokens, e.g. "[index3,middle3,thumb3]"')
    parser.add_argument('--contact_links_file', default="../data/contact_link.json", type=str,
                        help='JSON mapping of contact link presets (e.g., {"01": [...], "02": [...]})')
    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.5, type=float)
    parser.add_argument('--mu', default=0.98, type=float)
    parser.add_argument('--step_size', default=0.005, type=float)
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=18, type=float)
    parser.add_argument('--annealing_period', default=30, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--w_dis', default=100.0, type=float)
    parser.add_argument('--w_pen', default=100.0, type=float)
    parser.add_argument('--w_spen', default=10.0, type=float)
    parser.add_argument('--w_joints', default=1.0, type=float)
    parser.add_argument('--w_ground', default=300.0, type=float,
                        help='Weight for ground penetration penalty (dexhand021, non-random only).')
    # initialization settings
    parser.add_argument('--jitter_strength', default=0.1, type=float)
    parser.add_argument('--distance_lower', default=0.2, type=float)
    parser.add_argument('--distance_upper', default=0.3, type=float)
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    # dexhand021-specific init scaling (keeps ShadowHand defaults unchanged)
    parser.add_argument('--dexhand_distance_scale', default=1.3, type=float)
    parser.add_argument('--dexhand_theta_scale', default=0.7, type=float)
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.3, type=float)
    parser.add_argument('--thres_dis', default=0.005, type=float)
    parser.add_argument('--thres_pen', default=0.001, type=float)
    parser.add_argument('--max_e_pen', default=None, type=float,
                        help='If set, discard results with E_pen > max_e_pen when saving.')
    parser.add_argument('--object_num_samples', default=2000, type=int,
                        help='Number of object surface points used for E_pen (higher = more accurate, slower).')
    parser.add_argument('--random_obj_scale', action='store_true',
                        help='Enable random object scaling (default: off).')
    parser.add_argument('--random_hand', action='store_true',
                        help='Use randomized dexhand021 initialization (shadowhand-like) when set.')

    args = parser.parse_args()
    if args.name and "--result_path" not in sys.argv:
        args.result_path = os.path.join("..", "data", "experiments", args.name, "results")
    args.target_batch_size_each = args.batch_size_each
    args.batch_size_each = int(math.ceil(args.batch_size_each * 1.5))
    if args.batch_size_each != args.target_batch_size_each:
        print(f"Using oversampled batch_size_each={args.batch_size_each} (target={args.target_batch_size_each})")

    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f'gpu_list: {gpu_list}')

    # check whether arguments are valid and process arguments

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    contact_links_map = _load_contact_links_map(args.contact_links_file)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    if not os.path.exists(args.data_root_path):
        raise ValueError(f'data_root_path {args.data_root_path} doesn\'t exist')
    
    if args.todo:
        if args.object_code_list is not None or args.all:
            raise ValueError("when using --todo, do not set --object_code_list or --all")
    else:
        if (args.object_code_list is not None) + args.all != 1:
            raise ValueError('exactly one among \'object_code_list\' \'all\' should be specified')
    
    if args.todo:
        with open("todo.txt", "r") as f:
            lines = f.readlines()
            object_code_list_all = [line[:-1] for line in lines]
    else:
        object_code_list_all = os.listdir(args.data_root_path)
    
    if args.object_code_list is not None:
        object_code_list = args.object_code_list
        if len(object_code_list) == 1 and isinstance(object_code_list[0], str):
            s = object_code_list[0].strip()
            if s.startswith("[") and s.endswith("]"):
                import ast
                object_code_list = ast.literal_eval(s)
        if not set(object_code_list).issubset(set(object_code_list_all)):
            raise ValueError('object_code_list isn\'t a subset of dirs in data_root_path')
    else:
        object_code_list = object_code_list_all

    # Filter out objects missing required mesh files.
    valid_object_codes = []
    missing_mesh = []
    for object_code in object_code_list:
        mesh_path = os.path.join(args.data_root_path, object_code, "coacd", "decomposed.obj")
        if os.path.isfile(mesh_path):
            valid_object_codes.append(object_code)
        else:
            missing_mesh.append(object_code)
    if missing_mesh:
        print(f"Warning: {len(missing_mesh)} objects missing coacd/decomposed.obj; skipping.")
        print("  Examples:", ", ".join(missing_mesh[:10]))
    object_code_list = valid_object_codes
    
    if not args.overwrite:
        for object_code in object_code_list.copy():
            if os.path.exists(os.path.join(args.result_path, object_code + '.npy')):
                object_code_list.remove(object_code)

    if args.batch_size_each > args.max_total_batch_size:
        raise ValueError(f'batch_size_each {args.batch_size_each} should be smaller than max_total_batch_size {args.max_total_batch_size}')
    
    print(f'n_objects: {len(object_code_list)}')
    
    # generate

    random.seed(args.seed)
    random.shuffle(object_code_list)
    objects_each = args.max_total_batch_size // args.batch_size_each
    object_code_groups = [object_code_list[i: i + objects_each] for i in range(0, len(object_code_list), objects_each)]

    process_args = []
    for id, object_code_group in enumerate(object_code_groups):
        process_args.append((args, object_code_group, id + 1, gpu_list))

    with multiprocessing.Pool(len(gpu_list)) as p:
        it = tqdm(p.imap(generate, process_args), total=len(process_args), desc='generating', maxinterval=1000)
        list(it)
