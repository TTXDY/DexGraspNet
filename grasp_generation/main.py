"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: Entry of the program, generate small-scale experiments
"""

import os

if __file__ and os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d

from utils.hand_model import HandModel
from utils.hand_model_dexhand021 import HandModelDexHand021
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--hand_model_type', default='shadow_hand', type=str, choices=['shadow_hand', 'dexhand021'], help='Hand model type: shadow_hand or dexhand021')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="2", type=str)
parser.add_argument('--object_code_list', default=
    [
        'sem-Car-2f28e2bd754977da8cfac9da0ff28f62',
        'sem-Car-27e267f0570f121869a949ac99a843c4',
        'sem-Car-669043a8ce40d9d78781f76a6db4ab62',
        'sem-Car-58379002fbdaf20e61a47cff24512a0',
        'sem-Car-aeeb2fb31215f3249acee38782dd9680',
    ])
parser.add_argument('--name', default='exp_2', type=str)
parser.add_argument('--n_contact', default=4, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--n_iter', default=6000, type=int)
parser.add_argument('--contact_links', default=None, type=str,
                    help='Override contact selection with link tokens, e.g. "[index3,middle3,thumb3]"')
# hyper parameters (** Magic, don't touch! **)
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
parser.add_argument('--object_num_samples', default=2000, type=int,
                    help='Number of object surface points used for E_pen (higher = more accurate, slower).')

args = parser.parse_args()

# Fix object_code_list if it's a string from command line
if isinstance(args.object_code_list, str):
    import ast
    args.object_code_list = ast.literal_eval(args.object_code_list)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def _parse_contact_links(text):
    if text is None:
        return None
    s = text.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    s = s.replace('"', '').replace("'", '')
    tokens = [t.strip() for t in s.split(',') if t.strip()]
    return tokens or None


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


# prepare models

total_batch_size = len(args.object_code_list) * args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)

# Initialize hand model based on type
if args.hand_model_type == 'dexhand021':
    hand_model = HandModelDexHand021(
        mjcf_path='mjcf_dexhand021/dexhand021_right_simplified_floating_cleaned.xml',
        mesh_path='mjcf_dexhand021/meshes',
        contact_points_path='mjcf_dexhand021/contact_points.json',
        penetration_points_path='mjcf_dexhand021/penetration_points.json',
        device=device
    )
    print(f'Using dexhand021 hand model ({hand_model.n_dofs} DOF)')
else:
    hand_model = HandModel(
        mjcf_path='mjcf/shadow_hand_wrist_free.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device
    )
    print(f'Using Shadow Hand model ({hand_model.n_dofs} DOF)')

object_model = ObjectModel(
    data_root_path='../data/meshdata',
    batch_size_each=args.batch_size,
    num_samples=args.object_num_samples,
    device=device
)
object_model.initialize(args.object_code_list)

initialize_convex_hull(hand_model, object_model, args)

print('n_contact_candidates', hand_model.n_contact_candidates)
print('total batch size', total_batch_size)

contact_tokens = _parse_contact_links(args.contact_links)
if contact_tokens is not None:
    link_indices = []
    selected_links = []
    for token in contact_tokens:
        link_name = _map_contact_token(token, args.hand_model_type)
        if link_name not in hand_model.link_name_to_link_index:
            raise ValueError(f"Link '{link_name}' not found in hand model.")
        selected_links.append(link_name)
        link_idx = hand_model.link_name_to_link_index[link_name]
        candidate_indices = (hand_model.global_index_to_link_index == link_idx).nonzero(as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            raise ValueError(f"No contact candidates found for link '{link_name}'.")
        link_indices.append(candidate_indices)
    # Override n_contact to match provided tokens
    args.n_contact = len(contact_tokens)
    # Sample indices per token per batch
    idx_per_token = []
    for candidate_indices in link_indices:
        rand_idx = torch.randint(0, candidate_indices.numel(), (total_batch_size,), device=device)
        idx_per_token.append(candidate_indices[rand_idx])
    contact_point_indices = torch.stack(idx_per_token, dim=1)
    hand_model.set_parameters(hand_model.hand_pose, contact_point_indices)
    hand_model.allowed_contact_indices_per_slot = link_indices
    print(f"Selected contact links (tokens): {selected_links}")

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

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'logs'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
logger_config = {
    'thres_fc': args.thres_fc,
    'thres_dis': args.thres_dis,
    'thres_pen': args.thres_pen
}
logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)


# log settings

with open(os.path.join('../data/experiments', args.name, 'output.txt'), 'w') as f:
    f.write(str(args) + '\n')


# optimize

weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_spen=args.w_spen,
    w_joints=args.w_joints,
)
energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, 0, show=False)

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
    s = optimizer.try_step()

    optimizer.zero_grad()
    new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    new_energy.sum().backward(retain_graph=True)

    with torch.no_grad():
        accept, t = optimizer.accept_step(energy, new_energy)

        energy[accept] = new_energy[accept]
        E_dis[accept] = new_E_dis[accept]
        E_fc[accept] = new_E_fc[accept]
        E_pen[accept] = new_E_pen[accept]
        E_spen[accept] = new_E_spen[accept]
        E_joints[accept] = new_E_joints[accept]

        logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, step, show=False)


# save results
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']

joint_names = hand_model.joints_names
if len(joint_names) != hand_model.n_dofs:
    raise ValueError(f"Joint name count ({len(joint_names)}) does not match DOF ({hand_model.n_dofs}).")
try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
result_path = os.path.join('../data/experiments', args.name, 'results')
os.makedirs(result_path, exist_ok=True)
full_hand_pose = hand_model.hand_pose.detach()
full_contact_indices = hand_model.contact_point_indices.detach() if hand_model.contact_point_indices is not None else None
for i in range(len(args.object_code_list)):
    data_list = []
    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()
        surface_points = (object_model.surface_points_tensor[idx] * scale).detach().cpu().tolist()
        contact_point_indices = full_contact_indices[idx].detach().cpu().tolist() if full_contact_indices is not None else None
        hand_pose_cpu = full_hand_pose[idx].detach().cpu()
        hand_pose_raw = hand_pose_cpu.clone()
        qpos = dict(zip(joint_names, hand_pose_cpu[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose_cpu[3:9].unsqueeze(0))[0]
        translation = hand_pose_cpu[:3]
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos.update(dict(zip(rot_names, euler)))
        qpos.update(dict(zip(translation_names, translation.tolist())))
        hand_pose_st_cpu = hand_pose_st[idx].detach().cpu()
        qpos_st = dict(zip(joint_names, hand_pose_st_cpu[9:].tolist()))
        rot = robust_compute_rotation_matrix_from_ortho6d(hand_pose_st_cpu[3:9].unsqueeze(0))[0]
        translation = hand_pose_st_cpu[:3]
        euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
        qpos_st.update(dict(zip(rot_names, euler)))
        qpos_st.update(dict(zip(translation_names, translation.tolist())))
        # Recompute E_pen using saved surface points for consistency.
        with torch.no_grad():
            surf = torch.tensor(surface_points, dtype=torch.float, device=device).unsqueeze(0)
            hand_pose_raw_device = hand_pose_raw.to(device)
            hand_model.set_parameters(hand_pose_raw_device.unsqueeze(0))
            E_pen_recomputed = float(hand_model.cal_distance(surf).clamp_min(0).sum().item())
            if full_contact_indices is not None:
                hand_model.set_parameters(full_hand_pose, full_contact_indices)
            else:
                hand_model.set_parameters(full_hand_pose)
        data_list.append(dict(
            scale=scale,
            object_surface_points=surface_points,
            hand_pose_raw=hand_pose_raw.tolist(),
            qpos=qpos,
            qpos_st=qpos_st,
            contact_point_indices=contact_point_indices,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_pen_recomputed=E_pen_recomputed,
            E_spen=E_spen[idx].item(),
            E_joints=E_joints[idx].item(),
        ))
    np.save(os.path.join(result_path, args.object_code_list[i] + '.npy'), data_list, allow_pickle=True)
