"""
Default joint angles for dexhand021 initialization
Similar to Shadow Hand's joint_angles_mu, defines pre-bent finger posture for grasp initialization
"""

import torch
import numpy as np

# dexhand021 has 20 DOF (5 fingers × 4 joints)
# Joint order: r_f_joint1_1, r_f_joint1_2, r_f_joint1_3, r_f_joint1_4,
#              r_f_joint2_1, r_f_joint2_2, r_f_joint2_3, r_f_joint2_4,
#              r_f_joint3_1, r_f_joint3_2, r_f_joint3_3, r_f_joint3_4,
#              r_f_joint4_1, r_f_joint4_2, r_f_joint4_3, r_f_joint4_4,
#              r_f_joint5_1, r_f_joint5_2, r_f_joint5_3, r_f_joint5_4
#
# Finger order (confirmed): 1=Thumb, 2=Index, 3=Middle, 4=Ring, 5=Pinky

# Joint ranges from MJCF:
# Finger 1 (Thumb):   joint1_1 [0, 2.2],  joint1_2/3/4 [0, 1.3]
# Finger 2 (Index):   joint2_1 [0, 0.3],  joint2_2/3/4 [0, 1.3]
# Finger 3 (Middle):  joint3_1 [-0.001, 0.001], joint3_2/3/4 [0, 1.3]
# Finger 4 (Ring):    joint4_1 [0, 0.3],  joint4_2/3/4 [0, 1.3]
# Finger 5 (Pinky):   joint5_1 [0, 0.6],  joint5_2/3/4 [0, 1.3]

# Strategy: Use ~50% of max range for pre-bending
# - joint_1: Small spread (abduction/adduction)
# - joint_2/3/4: Moderate flexion (~0.6-0.7 rad, similar to Shadow Hand)

joint_angles_mu_dexhand021 = torch.tensor([
    # Finger 1 (Thumb) - 拇指
    np.deg2rad(50),  # joint1_1: thumb spread set to 50 deg for initialization
    0.8,   # joint1_2: more flexion (thumb needs more bend)
    0.8,   # joint1_3: more flexion
    0.7,   # joint1_4: flexion

    # Finger 2 (Index) - 食指
    0.2,   # joint2_1: slight spread (0.2 / 0.3 = 67%)
    0.6,   # joint2_2: flexion
    0.7,   # joint2_3: flexion
    0.6,   # joint2_4: flexion

    # Finger 3 (Middle) - 中指
    0.0,   # joint3_1: no spread (range is [-0.001, 0.001], almost fixed)
    0.6,   # joint3_2: flexion
    0.7,   # joint3_3: flexion
    0.6,   # joint3_4: flexion

    # Finger 4 (Ring) - 无名指
    0.0,   # joint4_1: no spread
    0.6,   # joint4_2: flexion
    0.7,   # joint4_3: flexion
    0.6,   # joint4_4: flexion

    # Finger 5 (Pinky) - 小指
    -0.1,  # joint5_1: slight inward (negative spread)
    0.6,   # joint5_2: flexion
    0.7,   # joint5_3: flexion
    0.6,   # joint5_4: flexion
], dtype=torch.float)

# Joint limits (from MJCF)
joints_lower_dexhand021 = torch.tensor([
    0.0, 0.0, 0.0, 0.0,      # Finger 1 (Thumb)
    0.0, 0.0, 0.0, 0.0,      # Finger 2 (Index)
    -0.001, 0.0, 0.0, 0.0,   # Finger 3 (Middle)
    0.0, 0.0, 0.0, 0.0,      # Finger 4 (Ring)
    0.0, 0.0, 0.0, 0.0,      # Finger 5 (Pinky)
], dtype=torch.float)

joints_upper_dexhand021 = torch.tensor([
    2.2, 1.3, 1.3, 1.3,      # Finger 1 (Thumb)
    0.3, 1.3, 1.3, 1.3,      # Finger 2 (Index)
    0.001, 1.3, 1.3, 1.3,    # Finger 3 (Middle)
    0.3, 1.3, 1.3, 1.3,      # Finger 4 (Ring)
    0.6, 1.3, 1.3, 1.3,      # Finger 5 (Pinky)
], dtype=torch.float)

if __name__ == "__main__":
    print("=" * 80)
    print("DEXHAND021 DEFAULT JOINT ANGLES")
    print("=" * 80)
    print(f"\nTotal DOF: {len(joint_angles_mu_dexhand021)}")
    print(f"\nDefault angles (radians):")
    print(f"  Finger 1 (Thumb):  {joint_angles_mu_dexhand021[0:4].tolist()}")
    print(f"  Finger 2 (Index):  {joint_angles_mu_dexhand021[4:8].tolist()}")
    print(f"  Finger 3 (Middle): {joint_angles_mu_dexhand021[8:12].tolist()}")
    print(f"  Finger 4 (Ring):   {joint_angles_mu_dexhand021[12:16].tolist()}")
    print(f"  Finger 5 (Pinky):  {joint_angles_mu_dexhand021[16:20].tolist()}")

    print(f"\nDefault angles (degrees):")
    print(f"  Finger 1 (Thumb):  {(joint_angles_mu_dexhand021[0:4] * 180 / np.pi).tolist()}")
    print(f"  Finger 2 (Index):  {(joint_angles_mu_dexhand021[4:8] * 180 / np.pi).tolist()}")
    print(f"  Finger 3 (Middle): {(joint_angles_mu_dexhand021[8:12] * 180 / np.pi).tolist()}")
    print(f"  Finger 4 (Ring):   {(joint_angles_mu_dexhand021[12:16] * 180 / np.pi).tolist()}")
    print(f"  Finger 5 (Pinky):  {(joint_angles_mu_dexhand021[16:20] * 180 / np.pi).tolist()}")

    print("\n" + "=" * 80)
