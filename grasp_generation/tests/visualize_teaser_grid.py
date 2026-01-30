"""
Teaser Grid Visualization Script
Generates a publication-quality grid visualization showing multiple grasped objects.
"""

import os
import sys
import argparse
import numpy as np
import torch
import transforms3d
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Get the script's directory and parent directory (grasp_generation/)
script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
parent_dir = os.path.dirname(script_dir)

# Change to parent directory and add to path
if os.path.exists(parent_dir):
    os.chdir(parent_dir)
    sys.path.insert(0, parent_dir)

from utils.hand_model_dexhand021 import HandModelDexHand021
from utils.object_model import ObjectModel

# Object list and grid layout
OBJECT_LIST = [
    'cube1', 'cube2', 'cuboid1', 'cuboid2', 'cuboid3',
    'cylinder1', 'cylinder2', 'cylinder3', 'cylinder5', 'cylinder6',
    'sphere1', 'sphere2', 'banana'
]

# Category-based color scheme (colorblind-friendly)
CATEGORY_COLORS = {
    'cube': '#E74C3C',      # Red
    'cuboid': '#3498DB',    # Blue
    'cylinder': '#2ECC71',  # Green
    'sphere': '#F39C12',    # Orange
    'other': '#9B59B6'      # Purple (banana)
}

def get_object_category(object_code):
    """Determine object category from object code."""
    if object_code.startswith('cube') and 'cuboid' not in object_code:
        return 'cube'
    elif object_code.startswith('cuboid'):
        return 'cuboid'
    elif object_code.startswith('cylinder'):
        return 'cylinder'
    elif object_code.startswith('sphere'):
        return 'sphere'
    else:
        return 'other'

def load_grasp_data(result_path, object_code, grasp_index=0):
    """Load grasp data from .npy file."""
    npy_path = os.path.join(result_path, f'{object_code}.npy')
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Result file not found: {npy_path}")

    data = np.load(npy_path, allow_pickle=True)
    if grasp_index >= len(data):
        raise IndexError(f"Grasp index {grasp_index} out of range (max: {len(data)-1})")

    return data[grasp_index]

def convert_qpos_to_hand_pose(data_dict, device):
    """Convert qpos dict to hand_pose tensor (following visualize_result.py pattern)."""
    # Translation and rotation names
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    control_names = HandModelDexHand021.CONTROL_NAMES

    # Check if hand_pose_raw exists (direct format)
    if 'hand_pose_raw' in data_dict:
        return torch.tensor(data_dict['hand_pose_raw'], dtype=torch.float, device=device)

    # Convert from qpos dict
    qpos = data_dict['qpos']

    # Extract translation
    translation = [qpos[name] for name in translation_names]

    # Extract rotation and convert to rot6d
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot6d = rot[:, :2].T.ravel().tolist()

    # Extract controls
    if 'controls' in data_dict:
        controls_entry = data_dict['controls']
        if isinstance(controls_entry, dict):
            controls = [controls_entry[name] for name in control_names]
        else:
            controls = list(controls_entry)
    elif all(name in qpos for name in control_names):
        controls = [qpos[name] for name in control_names]
    else:
        # Reconstruct from joint angles
        def get(name, default=0.0):
            return qpos.get(name, default)

        if all(name in qpos for name in ['r_f_joint2_1', 'r_f_joint4_1', 'r_f_joint5_1']):
            spread = (get('r_f_joint2_1') + get('r_f_joint4_1') + 0.5 * get('r_f_joint5_1')) / 3.0
        else:
            spread = get('r_f_joint2_1')

        controls = [
            get('r_f_joint1_1'),
            get('r_f_joint1_2'),
            get('r_f_joint1_3'),
            spread,
            get('r_f_joint2_2'),
            get('r_f_joint2_3'),
            get('r_f_joint3_2'),
            get('r_f_joint3_3'),
            get('r_f_joint4_2'),
            get('r_f_joint4_3'),
            get('r_f_joint5_2'),
            get('r_f_joint5_3'),
        ]

    # Combine: translation(3) + rot6d(6) + controls(12) = 21
    hand_pose = torch.tensor(translation + rot6d + controls, dtype=torch.float, device=device)
    return hand_pose

def render_grasp(hand_model, object_model, data_dict, object_code, device):
    """Render a single grasp and return Plotly traces."""
    # Convert pose
    hand_pose = convert_qpos_to_hand_pose(data_dict, device)

    # Set hand parameters
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    # Initialize object
    object_model.initialize(object_code)

    # Override scale with actual scale from saved data
    scale = data_dict.get('scale', 1.0)
    object_model.object_scale_tensor = torch.tensor(
        scale, dtype=torch.float, device=device
    ).reshape(1, 1)

    # Get Plotly data
    hand_data = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue')

    # Get object color based on category
    category = get_object_category(object_code)
    object_color = CATEGORY_COLORS[category]
    object_data = object_model.get_plotly_data(i=0, color=object_color, opacity=0.9)

    return hand_data + object_data

def create_teaser_grid(result_path, meshdata_path, output_html, output_png, device='cuda:0'):
    """Create teaser grid visualization."""
    print(f"Creating teaser grid visualization...")
    print(f"  Result path: {result_path}")
    print(f"  Meshdata path: {meshdata_path}")
    print(f"  Device: {device}")

    # Initialize models
    hand_model = HandModelDexHand021(
        mjcf_path='mjcf_dexhand021/dexhand021_right_simplified_floating_cleaned.xml',
        mesh_path='mjcf_dexhand021/meshes',
        contact_points_path='mjcf_dexhand021/contact_points.json',
        penetration_points_path='mjcf_dexhand021/penetration_points.json',
        n_surface_points=2000,
        device=device
    )

    object_model = ObjectModel(
        data_root_path=meshdata_path,
        batch_size_each=1,
        num_surface_samples=2000,
        device=device
    )

    # Create subplot grid (3 rows × 5 cols)
    fig = make_subplots(
        rows=3, cols=5,
        specs=[[{'type': 'scene'}] * 5 for _ in range(3)],
        subplot_titles=OBJECT_LIST + ['', ''],
        horizontal_spacing=0.02,
        vertical_spacing=0.05
    )

    # Render each object
    for idx, object_code in enumerate(OBJECT_LIST):
        row = idx // 5 + 1
        col = idx % 5 + 1

        print(f"\n[{idx+1}/{len(OBJECT_LIST)}] Rendering {object_code} (row={row}, col={col})...")

        try:
            # Load grasp data (best grasp = index 0)
            data_dict = load_grasp_data(result_path, object_code, grasp_index=0)
            print(f"  Energy: {data_dict.get('energy', 'N/A'):.3f}")

            # Render grasp
            traces = render_grasp(hand_model, object_model, data_dict, object_code, device)

            # Add traces to subplot
            for trace in traces:
                fig.add_trace(trace, row=row, col=col)

            print(f"  ✓ Added {len(traces)} traces to subplot")

        except Exception as e:
            print(f"  ✗ Error rendering {object_code}: {e}")
            continue

    # Configure camera and layout for all scenes
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )

    scene_layout = dict(
        camera=camera,
        aspectmode='data',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )

    # Update all scenes
    for idx in range(15):  # 3×5 = 15 subplots
        scene_name = 'scene' if idx == 0 else f'scene{idx+1}'
        fig.update_layout(**{scene_name: scene_layout})

    # Global layout
    fig.update_layout(
        title=dict(
            text="DexGraspNet: Teaser Grid Visualization",
            x=0.5,
            xanchor='center',
            font=dict(size=24)
        ),
        showlegend=False,
        width=3000,
        height=1000,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Save HTML
    print(f"\nSaving HTML to {output_html}...")
    fig.write_html(output_html)
    print(f"  ✓ HTML saved ({os.path.getsize(output_html) / 1024 / 1024:.2f} MB)")

    # Save PNG (requires kaleido)
    if output_png:
        try:
            print(f"\nSaving PNG to {output_png}...")
            fig.write_image(output_png, width=3000, height=1000)
            print(f"  ✓ PNG saved ({os.path.getsize(output_png) / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"  ✗ PNG export failed: {e}")
            print(f"  Hint: Install kaleido with: pip install kaleido")

    print("\n✓ Teaser grid visualization complete!")
    return fig

def main():
    parser = argparse.ArgumentParser(description='Generate teaser grid visualization')
    parser.add_argument('--result_path', type=str,
                        default='../data/experiments/dexhand021_grasping/results',
                        help='Path to results directory')
    parser.add_argument('--meshdata_path', type=str,
                        default='../data/meshdata',
                        help='Path to meshdata directory')
    parser.add_argument('--output_html', type=str,
                        default='../teaser_visualization.html',
                        help='Output HTML file path')
    parser.add_argument('--output_png', type=str,
                        default='../teaser_visualization.png',
                        help='Output PNG file path (requires kaleido)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--no_png', action='store_true',
                        help='Skip PNG export')

    args = parser.parse_args()

    # Convert to absolute paths
    args.result_path = os.path.abspath(args.result_path)
    args.meshdata_path = os.path.abspath(args.meshdata_path)
    args.output_html = os.path.abspath(args.output_html)
    if not args.no_png:
        args.output_png = os.path.abspath(args.output_png)
    else:
        args.output_png = None

    # Create visualization
    create_teaser_grid(
        result_path=args.result_path,
        meshdata_path=args.meshdata_path,
        output_html=args.output_html,
        output_png=args.output_png,
        device=args.device
    )

if __name__ == '__main__':
    main()
