"""
Recenter saved grasp results by moving object surface points to the origin and
shifting hand poses by the same offset.

Example:
  /home/jay/anaconda3/envs/dexgraspnet-dexhand/bin/python scripts/recenter_results.py \
    --input ../data/experiments/dexhand021_grasping/results/cylinder1.npy \
    --output ../data/experiments/dexhand021_grasping/results/cylinder1_centered.npy
"""

import argparse
import os
import sys

import numpy as np


def _compute_center(points, mode):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"object_surface_points must be (N,3), got {pts.shape}")
    if mode == "bbox":
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        return (mins + maxs) * 0.5
    return pts.mean(axis=0)


def _shift_hand_pose_raw(hand_pose_raw, offset):
    if hand_pose_raw is None:
        return None
    pose = np.asarray(hand_pose_raw, dtype=np.float32).copy()
    if pose.shape[0] < 3:
        return hand_pose_raw
    pose[:3] -= offset
    return pose.tolist()


def _shift_hand_pose_3_3_12(hand_pose_3_3_12, offset):
    if hand_pose_3_3_12 is None:
        return None
    pose = np.asarray(hand_pose_3_3_12, dtype=np.float32).copy()
    if pose.shape[0] < 3:
        return hand_pose_3_3_12
    pose[:3] -= offset
    return pose.tolist()


def _shift_qpos(qpos, offset):
    if qpos is None:
        return None
    qpos = dict(qpos)
    for name, idx in zip(["WRJTx", "WRJTy", "WRJTz"], range(3)):
        if name in qpos:
            qpos[name] = float(qpos[name]) - float(offset[idx])
    return qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .npy file")
    parser.add_argument("--output", required=True, help="Output .npy file")
    parser.add_argument("--center_mode", choices=["mean", "bbox"], default="mean",
                        help="Center from surface points: mean or bbox center")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    data_arr = np.load(args.input, allow_pickle=True)
    if len(data_arr) == 0:
        raise ValueError("Empty result file")

    new_arr = []
    for data_dict in data_arr:
        if "object_surface_points" not in data_dict:
            raise KeyError("object_surface_points missing; cannot recenter")
        center = _compute_center(data_dict["object_surface_points"], args.center_mode)

        new_dict = dict(data_dict)
        new_dict["object_surface_points"] = (np.asarray(data_dict["object_surface_points"], dtype=np.float32) - center).tolist()
        new_dict["hand_pose_raw"] = _shift_hand_pose_raw(data_dict.get("hand_pose_raw"), center)
        new_dict["intrinsic_euler_hand_pose_3_3_12"] = _shift_hand_pose_3_3_12(
            data_dict.get("intrinsic_euler_hand_pose_3_3_12"), center
        )
        new_dict["qpos"] = _shift_qpos(data_dict.get("qpos"), center)
        new_dict["qpos_st"] = _shift_qpos(data_dict.get("qpos_st"), center)

        new_arr.append(new_dict)

    np.save(args.output, np.array(new_arr, dtype=object), allow_pickle=True)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
