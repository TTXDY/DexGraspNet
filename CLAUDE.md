# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DexGraspNet is a grasp generation system that synthesizes dexterous hand grasps for 3D objects using optimization-based methods. This repository uses **Shadow Hand** and **DexHand021** robot models and performs physics-based grasp optimization through simulated annealing.

**Key Difference from MANO branch**: This repository uses robotic hands (MJCF/URDF) instead of MANO (parametric human hand model).

## Quick Start Guide

### Prerequisites
- Linux system with NVIDIA GPU
- CUDA 12.1 compatible GPU
- Python 3.8
- Conda package manager

### Complete Installation & Deployment

#### 1. Environment Setup

```bash
# Create conda environment
conda create -n dexgraspnet python=3.8 -y
conda activate dexgraspnet

# Install basic dependencies
pip install numpy tqdm transforms3d trimesh plotly tensorboard

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install CUDA toolkit (required for compiling extensions)
conda install -c nvidia cuda-toolkit=12.1 -y
conda install -c conda-forge libxcrypt -y

# Install pytorch3d (MUST be built from source for GPU support)
pip uninstall pytorch3d -y  # Remove any existing installation
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install additional required packages
pip install urdf_parser_py scipy networkx
conda install rtree -y

# Install pytorch_kinematics (modified, included in repo)
cd thirdparty/pytorch_kinematics
pip install -e .
cd ../..

# Install TorchSDF (requires compatibility fixes for PyTorch 2.x)
cd thirdparty/TorchSDF
git checkout 0.1.0
export IGNORE_TORCH_VER=1
python setup.py develop
cd ../..
```

**CRITICAL**: pytorch3d MUST be installed with `--no-build-isolation` and built from source to enable GPU support. Pre-built wheels do not include CUDA kernels.

**Important Notes**:
- Python 3.8 is recommended (Python 3.7 only needed if using Isaac Gym validation)
- PyTorch 2.1.0 + CUDA 12.1 is the recommended version
- TorchSDF requires compatibility fixes for PyTorch 2.x (already applied in this repo)

#### 2. Build External Tools (for asset_process)

```bash
cd asset_process/

# Build ManifoldPlus
git clone https://github.com/hjwdzh/ManifoldPlus.git
cd ManifoldPlus
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
cd ../..

# Build CoACD
git clone --recurse-submodules https://github.com/SarahWeiii/CoACD.git
cd CoACD
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
cd ../..
```

#### 3. Process 3D Models

**Option A: Use existing STL/OBJ models**

If you have your own 3D models in `data/raw_models/`:

```bash
cd asset_process/

# Step 1: Convert to manifold meshes
# Note: ManifoldPlus may crash on some models. Use trimesh as fallback:
python convert_with_trimesh.py --src ../data/raw_models --dst ../data/manifolds

# Step 2: Normalize models
python normalize.py --src ../data/manifolds --dst ../data/normalized_models

# Step 3: Convex decomposition
python decompose_list.py --src ../data/normalized_models --dst ../data/meshdata --coacd_path ./CoACD/build/main
bash run.sh  # or: python poolrun.py -p 32 for parallel processing
```

**Optional: Real-scale meshes**

The normalization step rescales objects to a canonical size. If you have real-world target sizes, scale normalized meshes with:

```bash
python scale_from_json.py --src ../data/normalized_models --dst ../data/real_scale_models --size_json path/to/object_real_size.json
```

`object_real_size.json` should contain per-object bounding-box sizes in meters (xyz extents).

**Option B: Extract from ShapeNet datasets**

If you have ShapeNet datasets:

```bash
cd asset_process/

# Extract from ShapeNetCore
python extract.py --src data/ShapeNetCore.v2 --dst data/raw_models --set core

# Then follow Option A steps above
```

#### 4. Generate Grasps

```bash
cd grasp_generation/

# Generate grasps for one object
python main.py \
  --object_code_list "['banana']" \
  --name test_shadowhand \
  --n_contact 4 \
  --batch_size 128 \
  --n_iter 6000 \
  --gpu "0"

# Generate grasps for multiple objects
python main.py \
  --object_code_list "['banana', 'cube1', 'sphere1']" \
  --name test_multiple \
  --n_contact 4 \
  --batch_size 128 \
  --n_iter 6000 \
  --gpu "0"
```

#### 5. Visualize Results

```bash
cd grasp_generation/tests/

# Visualize specific grasp (generates HTML file)
python visualize_result.py \
  --object_code banana \
  --num 0 \
  --result_path ../data/experiments/test_shadowhand/results

# The HTML file will be saved in grasp_generation/ directory
# Open grasp_visualization_banana_0.html in a web browser
```

**Note**: `--num` is the grasp index (0 = best grasp, 1 = second best, etc.). Each visualization shows ONE grasp.

## Common Issues & Solutions

### Issue 1: TorchSDF setup.py - IGNORE_TORCH_VER not defined

**Problem**: 
```
NameError: name 'IGNORE_TORCH_VER' is not defined
```

**Cause**: TorchSDF's setup.py uses `IGNORE_TORCH_VER` variable but doesn't read it from environment.

**Solution**: Already fixed in this repository. The fix adds this line to `thirdparty/TorchSDF/setup.py` after line 16:
```python
IGNORE_TORCH_VER = os.environ.get('IGNORE_TORCH_VER', '0') == '1'
```

### Issue 2: TorchSDF compilation - CHECK_EQ undefined

**Problem**:
```
error: identifier "CHECK_EQ" is undefined
```

**Cause**: PyTorch 2.x removed the `CHECK_EQ` macro that TorchSDF uses.

**Solution**: Already fixed in this repository. The fix modifies `thirdparty/TorchSDF/torchsdf/csrc/utils.h` line 22-27:
```cpp
// OLD (PyTorch 1.x):
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

// NEW (PyTorch 2.x):
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    TORCH_CHECK(error == cudaSuccess, "CUDA error: ", cudaGetErrorString(error)); \
  } while (0)
```

### Issue 3: main.py - __file__ is empty string

**Problem**:
```
FileNotFoundError: [Errno 2] No such file or directory: ''
```

**Cause**: `__file__` variable can be empty in some execution contexts.

**Solution**: Already fixed in this repository. The fix modifies `grasp_generation/main.py` line 6-9:
```python
# OLD:
os.chdir(os.path.dirname(__file__))

# NEW:
if __file__ and os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))
```

### Issue 4: main.py - object_code_list parsing error

**Problem**:
```
ValueError: string is not a file: `../data/meshdata/[/coacd/decomposed.obj`
```

**Cause**: Command line argument `"['banana']"` is not parsed as Python list.

**Solution**: Already fixed in this repository. Two changes:
1. Remove `type=list` from argument definition (line 42)
2. Add parsing after `parser.parse_args()` (line 72-75):
```python
# Fix object_code_list if it's a string from command line
if isinstance(args.object_code_list, str):
    import ast
    args.object_code_list = ast.literal_eval(args.object_code_list)
```

### Issue 5: initializations.py - remove_degenerate_faces not found

**Problem**:
```
AttributeError: 'Trimesh' object has no attribute 'remove_degenerate_faces'
```

**Cause**: Trimesh API changed - `remove_degenerate_faces()` method was replaced with `nondegenerate_faces()` property.

**Solution**: Already fixed in this repository. The fix modifies `grasp_generation/utils/initializations.py` line 45-48:
```python
# OLD:
mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]

# NEW:
if hasattr(mesh_origin, 'nondegenerate_faces'):
    mesh_origin.update_faces(mesh_origin.nondegenerate_faces())
```

### Issue 6: visualize_result.py - utils module not found

**Problem**:
```
ModuleNotFoundError: No module named 'utils'
```

**Cause**: Script runs from `tests/` directory but `utils` is in parent directory.

**Solution**: Already fixed in this repository. The fix modifies `grasp_generation/tests/visualize_result.py` line 6-11:
```python
# Get the script's directory and parent directory (grasp_generation/)
script_dir = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()
parent_dir = os.path.dirname(script_dir)

# Change to parent directory and add to path
if os.path.exists(parent_dir):
    os.chdir(parent_dir)
    sys.path.insert(0, parent_dir)
```


## Repository Structure

```
DexGraspNet/
├── asset_process/           # 3D object mesh preprocessing pipeline
│   ├── extract.py          # Extract meshes from datasets
│   ├── convert_with_trimesh.py  # Convert to manifold (trimesh fallback)
│   ├── normalize.py        # Center, scale, filter meshes
│   ├── decompose_list.py   # Convex decomposition with CoACD
│   └── utils/              # Extraction utilities
├── grasp_generation/        # Grasp synthesis system
│   ├── main.py             # Entry point for experiments
│   ├── mjcf/               # Shadow Hand MJCF files
│   │   ├── shadow_hand_wrist_free.xml
│   │   ├── contact_points.json
│   │   └── penetration_points.json
│   ├── utils/              # Core modules
│   │   ├── hand_model.py   # Shadow Hand model (pytorch_kinematics)
│   │   ├── object_model.py # Object mesh loading and SDF
│   │   ├── energy.py       # Energy function (5 terms)
│   │   ├── optimizer.py    # Simulated annealing
│   │   ├── initializations.py  # Convex hull initialization
│   │   └── logger.py       # Training logging
│   └── tests/
│       └── visualize_result.py  # Grasp visualization
├── data/                    # Symlinked data directories (not in repo)
│   ├── raw_models/         # Input STL/OBJ files
│   ├── manifolds/          # Watertight meshes
│   ├── normalized_models/  # Normalized meshes
│   ├── meshdata/           # Processed objects with URDF
│   └── experiments/        # Experimental results
└── thirdparty/
    ├── pytorch_kinematics/  # Modified for speed (included)
    ├── TorchSDF/           # Custom Kaolin fork (clone separately)
    ├── CoACD/              # Convex decomposition (build separately)
    └── ManifoldPlus/       # Manifold conversion (build separately)
```

## Architecture Details

### Optimization Loop (main.py)

The core optimization uses simulated annealing with the following flow:

1. **Initialization** - Sample initial hand poses using convex hull method
2. **Try step** - Optimizer proposes new hand pose using RMSProp gradient descent and randomly resamples contact points
3. **Compute energy** - Calculate 5 energy terms for new configuration
4. **Accept/reject** - Use Metropolis criterion with temperature schedule
5. **Log** - Track energy components throughout optimization

### Energy Function (utils/energy.py)

Total energy = E_fc + w_dis×E_dis + w_pen×E_pen + w_joints×E_joints + w_spen×E_spen

- **E_fc**: Force closure quality (wrench space analysis)
- **E_dis**: Contact point distance to object surface
- **E_pen**: Object-to-hand penetration (sampled surface points)
- **E_joints**: Joint limit violations
- **E_spen**: Hand self-penetration

### Hand Model (utils/hand_model.py)

Wraps Shadow Hand MJCF model with:
- Forward kinematics via `pytorch_kinematics`
- Contact point management via pre-selected candidate indices
- SDF computation using KNN + vertex normals
- 24 DOF dexterous hand

**Key files**:
- `mjcf/shadow_hand_wrist_free.xml` - MJCF robot definition
- `mjcf/contact_points.json` - Hand-selected contact candidates
- `mjcf/penetration_points.json` - Keypoints for self-penetration check

### Object Model (utils/object_model.py)

Manages object meshes with:
- Random scale selection from [0.06, 0.08, 0.1]
- Surface point sampling via farthest point sampling (FPS)
- SDF computation using `torchsdf` library
- Batch processing for multiple objects

### Optimizer (utils/optimizer.py)

Custom simulated annealing implementation:
- RMSProp for continuous parameters (pose)
- Random resampling for discrete parameters (contact indices)
- Temperature-based acceptance with exponential decay
- Step size decay synchronized with temperature


## Data Paths

The system expects this directory structure:
```
DexGraspNet/
├── data/
│   ├── raw_models/            # Input STL/OBJ files
│   ├── manifolds/             # Watertight meshes
│   ├── normalized_models/     # Normalized meshes
│   ├── meshdata/              # Processed objects (output of asset_process)
│   │   └── {object_code}/
│   │       └── coacd/
│   │           ├── decomposed.obj
│   │           ├── coacd_convex_piece_*.obj
│   │           └── coacd.urdf
│   └── experiments/           # Grasp generation results
│       └── {exp_name}/
│           ├── logs/
│           ├── results/
│           └── output.txt
```

Note: `meshdata`, `experiments` are typically symlinks (see .gitignore).

## Command Reference

### Grasp Generation Parameters

```bash
python main.py \
  --object_code_list "['object1', 'object2']" \  # List of object codes
  --name experiment_name \                        # Experiment name
  --n_contact 4 \                                 # Number of contact points
  --batch_size 128 \                              # Grasps per object
  --n_iter 6000 \                                 # Optimization iterations
  --gpu "0"                                       # GPU device ID
```

**Key parameters**:
- `--object_code_list`: Python list string of object codes (from meshdata/)
- `--name`: Experiment name (results saved to ../data/experiments/{name}/)
- `--n_contact`: Number of contact points (default: 4)
- `--batch_size`: Number of grasp attempts per object (default: 128)
- `--n_iter`: Optimization iterations (default: 6000)
- `--gpu`: GPU device ID

**Hyperparameters** (marked as "Magic, don't touch!" in code):
- `--w_dis`, `--w_pen`, `--w_joints`, `--w_spen`: Energy term weights
- `--starting_temperature`, `--temperature_decay`, `--annealing_period`: Annealing schedule
- `--step_size`, `--stepsize_period`, `--mu`: RMSProp parameters
- `--switch_possibility`: Contact point resampling probability

### Visualization Parameters

```bash
python visualize_result.py \
  --object_code banana \                          # Object code
  --num 0 \                                       # Grasp index (0=best)
  --result_path ../data/experiments/test/results  # Results directory
```

**Note**: `--num` is the grasp INDEX (0-127 for batch_size=128), not the number of grasps to display. Lower index = better grasp quality.

## Result Format

Results saved as .npy files containing list of dicts:
```python
{
    'scale': float,
    'qpos': {'trans': [3], 'rot': [3], 'thetas': [n_dofs]},
    'contact_point_indices': [n_contact],
    'qpos_st': {...},  # Initial pose
    'energy': float,
    'E_fc': float,
    'E_dis': float,
    'E_pen': float,
    'E_joints': float,
    'E_spen': float
}
```

**qpos structure**:
- `WRJTx, WRJTy, WRJTz`: Translation (meters)
- `WRJRx, WRJRy, WRJRz`: Rotation (euler angles, xyz convention)
- `robot0:FFJ3`, `robot0:FFJ2`, ... : Joint angles for Shadow Hand

## Performance Notes

- Grasp generation runs at ~40-55 iterations/second on NVIDIA RTX 4060
- 6000 iterations takes approximately 2-3 minutes per object
- Asset processing (manifold + normalize + decompose) takes ~5-10 seconds per model
- Visualization HTML files are typically 100-500KB

## Testing Your Installation

Quick test to verify everything works:

```bash
# 1. Process a simple model
cd asset_process/
python convert_with_trimesh.py --src ../data/raw_models --dst ../data/manifolds
python normalize.py --src ../data/manifolds --dst ../data/normalized_models
python decompose_list.py --src ../data/normalized_models --dst ../data/meshdata --coacd_path ./CoACD/build/main
bash run.sh

# 2. Generate grasps
cd ../grasp_generation/
python main.py --object_code_list "['banana']" --name test --n_contact 4 --batch_size 32 --n_iter 1000 --gpu "0"

# 3. Visualize
cd tests/
python visualize_result.py --object_code banana --num 0 --result_path ../data/experiments/test/results

echo "Installation test complete! Check grasp_visualization_banana_0.html"
```

If all steps complete without errors, your installation is successful!

## Development Notes

### Batch Processing

The system processes multiple objects simultaneously:
- `total_batch_size = len(object_code_list) × batch_size`
- Each object gets `batch_size` grasp attempts
- Results are indexed as: `idx = object_index × batch_size + sample_index`

### Contact Point System

Contact points are selected from pre-defined candidates (contact_points.json):
- Indices reference vertices on the Shadow Hand mesh
- Optimizer randomly resamples indices during optimization
- Final contact points are stored per-grasp in results

### Differences from MANO Branch

**Hand Model**:
- MANO: Parametric human hand (45 DOF, manopth)
- Shadow Hand: Robotic hand (22 DOF, pytorch_kinematics + MJCF)

**Dependencies**:
- MANO: manopth, chumpy, numpy<1.24
- Shadow Hand: pytorch_kinematics, urdf_parser_py

**Key Insight**: Both branches can use Python 3.8 + PyTorch 2.1.0 without needing Isaac Gym for grasp generation.
