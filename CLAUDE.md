# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DexGraspNet is a grasp generation system that synthesizes dexterous hand grasps for 3D objects using optimization-based methods. The system uses the MANO hand model and performs physics-based grasp optimization through simulated annealing.

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

# Install manopth
pip install git+https://github.com/hassony2/manopth.git

# Install TorchSDF
cd /tmp
git clone https://github.com/wrc042/TorchSDF.git
cd TorchSDF
export IGNORE_TORCH_VER=1
python setup.py develop
cd -

# Install additional required packages
pip install "numpy<1.24"  # chumpy compatibility
pip install chumpy opencv-python lxml rtree scipy
```

**CRITICAL**: pytorch3d MUST be installed with `--no-build-isolation` and built from source to enable GPU support. Pre-built wheels do not include CUDA kernels.

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
  --name test_banana \
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
  --result_path ../data/experiments/test_banana/results

# The HTML file will be saved in grasp_generation/ directory
# Open grasp_visualization_banana_0.html in a web browser
```

**Note**: `--num` is the grasp index (0 = best grasp, 1 = second best, etc.). Each visualization shows ONE grasp.

## Repository Structure

The codebase is organized into two main modules:

### `asset_process/`
Preprocessing pipeline for 3D object models. Converts raw meshes from various datasets into simulation-ready assets.

**Pipeline stages** (must be run in order):
1. **Extraction** - Organize models from source datasets
2. **Manifold** - Convert to manifold meshes using ManifoldPlus (or trimesh as fallback)
3. **Normalization** - Center, scale, and filter models
4. **Decomposition** - Convex decomposition using CoACD, generates URDF files

**Supported Input Formats**: STL, OBJ, OFF

### `grasp_generation/`
Core grasp synthesis system using optimization.

**Key components**:
- `main.py` - Entry point for grasp generation experiments
- `utils/hand_model.py` - MANO hand model wrapper with contact point management
- `utils/object_model.py` - Object mesh loading, SDF computation, surface sampling
- `utils/energy.py` - Energy function with 5 terms: force closure (E_fc), contact distance (E_dis), penetration (E_pen), pose prior (E_prior), self-penetration (E_spen)
- `utils/optimizer.py` - Simulated annealing optimizer with RMSProp and contact resampling
- `utils/initializations.py` - Convex hull-based initialization
- `utils/logger.py` - Training logging and result tracking
- `tests/` - Visualization scripts for hand model, initialization, and results

## Common Issues & Solutions

### Issue 1: ManifoldPlus Segmentation Faults

**Problem**: ManifoldPlus crashes with segmentation faults on STL files.

**Solution**: Use the provided `convert_with_trimesh.py` script instead:
```bash
python convert_with_trimesh.py --src ../data/raw_models --dst ../data/manifolds
```

This uses trimesh's built-in mesh repair functions as a more stable alternative.

### Issue 2: pytorch3d "Not compiled with GPU support"

**Problem**: RuntimeError: Not compiled with GPU support.

**Solution**: pytorch3d must be built from source with CUDA support:
```bash
pip uninstall pytorch3d -y
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Pre-built wheels do not include CUDA kernels.

### Issue 3: chumpy NumPy compatibility

**Problem**: ImportError: cannot import name 'bool' from 'numpy'

**Solution**: Downgrade NumPy to <1.24:
```bash
pip install "numpy<1.24"
```

chumpy (required by manopth) is incompatible with NumPy 1.24+.

### Issue 4: trimesh API changes

**Problem**: AttributeError: 'Trimesh' object has no attribute 'remove_degenerate_faces'

**Solution**: The code has been updated to use `nondegenerate_faces()` property instead. If you encounter this, the fix is in `utils/initializations.py`:
```python
if hasattr(mesh_origin, 'nondegenerate_faces'):
    mesh_origin.update_faces(mesh_origin.nondegenerate_faces())
```

### Issue 5: Missing dependencies

**Problem**: ModuleNotFoundError for various packages.

**Solution**: Install all required packages:
```bash
pip install lxml chumpy opencv-python rtree scipy
```

### Issue 6: __file__ is empty string

**Problem**: FileNotFoundError: [Errno 2] No such file or directory: ''

**Solution**: The code has been updated to handle empty `__file__`. If you encounter this, add checks:
```python
if __file__ and os.path.dirname(__file__):
    os.chdir(os.path.dirname(__file__))
```

### Issue 7: Visualization doesn't open

**Problem**: Plotly visualization doesn't display on remote servers.

**Solution**: The visualization scripts now save HTML files:
```python
fig.write_html('grasp_visualization.html')
```

Download and open the HTML file in your local browser.

## Architecture Details

### Optimization Loop (main.py)

The core optimization uses simulated annealing with the following flow:

1. **Try step** - Optimizer proposes new hand pose using RMSProp gradient descent and randomly resamples contact points
2. **Compute energy** - Calculate 5 energy terms for new configuration
3. **Accept/reject** - Use Metropolis criterion with temperature schedule
4. **Log** - Track energy components throughout optimization

### Energy Function (utils/energy.py)

Total energy = E_fc + w_dis×E_dis + w_pen×E_pen + w_prior×E_prior + w_spen×E_spen

- **E_fc**: Force closure quality (wrench space analysis)
- **E_dis**: Contact point distance to object surface
- **E_pen**: Object-to-hand penetration (sampled surface points)
- **E_prior**: Hand pose prior (Gaussian distribution from MANO)
- **E_spen**: Hand self-penetration

### Hand Model (utils/hand_model.py)

Wraps MANO hand model with:
- 51-dimensional pose: [translation(3), rotation(3), joint_angles(45)]
- Contact point management via pre-selected candidate indices
- SDF computation using KNN + vertex normals
- Requires `manopth` library and MANO_RIGHT.pkl

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
├── asset_process/
│   ├── ManifoldPlus/
│   ├── CoACD/
│   └── convert_with_trimesh.py  # Fallback for ManifoldPlus
└── grasp_generation/
    ├── main.py
    ├── utils/
    ├── tests/
    └── mano/
        ��── MANO_RIGHT.pkl
        ├── contact_indices.json
        └── pose_distrib.pt
```

Note: `meshdata`, `experiments`, `graspdata`, and `dataset` are typically symlinks (see .gitignore).

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
- `--w_dis`, `--w_pen`, `--w_prior`, `--w_spen`: Energy term weights
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

## Development Notes

### Working with Energy Terms

When modifying energy functions, all terms must:
- Return tensors of shape (batch_size,)
- Be differentiable for gradient-based optimization
- Use consistent device placement (CPU/CUDA)

### Batch Processing

The system processes multiple objects simultaneously:
- `total_batch_size = len(object_code_list) × batch_size`
- Each object gets `batch_size` grasp attempts
- Results are indexed as: `idx = object_index × batch_size + sample_index`

### Contact Point System

Contact points are selected from pre-defined candidates (contact_indices.json):
- Indices reference vertices on the MANO hand mesh
- Optimizer randomly resamples indices during optimization
- Final contact points are stored per-grasp in results

### Result Format

Results saved as .npy files containing list of dicts:
```python
{
    'scale': float,
    'qpos': {'trans': [3], 'rot': [3], 'thetas': [45]},
    'contact_point_indices': [n_contact],
    'qpos_st': {...},  # Initial pose
    'energy': float,
    'E_fc': float,
    'E_dis': float,
    'E_pen': float,
    'E_prior': float,
    'E_spen': float
}
```

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
echo "Testing asset processing..."
python convert_with_trimesh.py --src ../data/raw_models --dst ../data/manifolds
python normalize.py --src ../data/manifolds --dst ../data/normalized_models
python decompose_list.py --src ../data/normalized_models --dst ../data/meshdata --coacd_path ./CoACD/build/main
bash run.sh

# 2. Generate grasps
cd ../grasp_generation/
echo "Testing grasp generation..."
python main.py --object_code_list "['banana']" --name test --n_contact 4 --batch_size 32 --n_iter 1000 --gpu "0"

# 3. Visualize
cd tests/
echo "Testing visualization..."
python visualize_result.py --object_code banana --num 0 --result_path ../data/experiments/test/results

echo "Installation test complete! Check grasp_visualization_banana_0.html"
```

If all steps complete without errors, your installation is successful!
