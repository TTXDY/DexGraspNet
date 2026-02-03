"""
Default joint angles for dexhand021 initialization
Similar to Shadow Hand's joint_angles_mu, defines pre-bent finger posture for grasp initialization
"""

import torch
import numpy as np

# dexhand021 control-space definition (12 inputs)
#
# Control order:
#   0  ctrl_thumb_spread
#   1  ctrl_thumb_mcp
#   2  ctrl_thumb_dip         -> joint1_3 + joint1_4
#   3  ctrl_finger_spread     -> joint2_1 + joint4_1 + (2x) joint5_1
#   4  ctrl_index_mcp
#   5  ctrl_index_dip         -> joint2_3 + joint2_4
#   6  ctrl_middle_mcp
#   7  ctrl_middle_dip        -> joint3_3 + joint3_4
#   8  ctrl_ring_mcp
#   9  ctrl_ring_dip          -> joint4_3 + joint4_4
#   10 ctrl_pinky_mcp
#   11 ctrl_pinky_dip         -> joint5_3 + joint5_4
#
# Control ranges from MJCF (radians):
#   thumb spread: [0, 2.2]
#   finger spread: [0, 0.3] (pinky joint5_1 moves 2x)
#   all MCP/DIP flexion: [0, 1.3]

# Strategy: use moderate flexion for pre-bending in control space.
joint_angles_mu_dexhand021 = torch.tensor([
    np.deg2rad(80),  # ctrl_thumb_spread
    0.0,            # ctrl_thumb_mcp
    0.1,            # ctrl_thumb_dip
    0.1,            # ctrl_finger_spread
    0.3,            # ctrl_index_mcp
    0.3,            # ctrl_index_dip
    0.3,            # ctrl_middle_mcp
    0.3,            # ctrl_middle_dip
    0.3,            # ctrl_ring_mcp
    0.3,            # ctrl_ring_dip
    0.3,            # ctrl_pinky_mcp
    0.3,            # ctrl_pinky_dip
], dtype=torch.float)

# Control limits
joints_lower_dexhand021 = torch.tensor([
    0.0, 0.0, 0.0, 0.0,  # thumb spread/mcp/dip + finger spread
    0.0, 0.0,           # index mcp/dip
    0.0, 0.0,           # middle mcp/dip
    0.0, 0.0,           # ring mcp/dip
    0.0, 0.0,           # pinky mcp/dip
], dtype=torch.float)

joints_upper_dexhand021 = torch.tensor([
    2.2, 1.3, 1.3, 0.3,
    1.3, 1.3,
    1.3, 1.3,
    1.3, 1.3,
    1.3, 1.3,
], dtype=torch.float)

if __name__ == "__main__":
    print("=" * 80)
    print("DEXHAND021 DEFAULT JOINT ANGLES")
    print("=" * 80)
    print(f"\nTotal DOF (controls): {len(joint_angles_mu_dexhand021)}")
    print(f"\nDefault control angles (radians):")
    print(joint_angles_mu_dexhand021.tolist())

    print(f"\nDefault control angles (degrees):")
    print((joint_angles_mu_dexhand021 * 180 / np.pi).tolist())

    print("\n" + "=" * 80)
