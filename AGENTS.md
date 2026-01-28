# Repository Guidelines

## Project Structure & Module Organization
- `asset_process/`: preprocessing pipeline for object meshes (extract, manifold, normalize, decompose).
- `grasp_generation/`: grasp synthesis code, scripts, and visualization utilities.
- `data/`: expected runtime data layout (often symlinked), including `raw_models/`, `meshdata/`, `experiments/`.
- `thirdparty/`: external dependencies like `pytorch_kinematics`, `TorchSDF`, `CoACD`, `ManifoldPlus`.
- `images/`: documentation assets used by the README.

## Build, Test, and Development Commands
- Create environment (Python 3.8 preferred; 3.7 if using Isaac Gym):
  - `conda create -n dexgraspnet python=3.8 && conda activate dexgraspnet`
- Install core deps (example): `pip install numpy tqdm transforms3d trimesh plotly tensorboard`.
- Install modified kinematics: `cd thirdparty/pytorch_kinematics && pip install -e .`.
- Asset processing pipeline (from `asset_process/`):
  - `python extract.py --src <dataset> --dst ../data/raw_models --set <set>`
  - `python manifold.py --src ../data/raw_models --dst ../data/manifolds --manifold_path ../thirdparty/ManifoldPlus/build/manifold`
  - `python normalize.py --src ../data/manifolds --dst ../data/normalized_models`
  - `python decompose_list.py --src ../data/normalized_models --dst ../data/meshdata --coacd_path ../thirdparty/CoACD/build/main`
- Grasp generation (from `grasp_generation/`):
  - `python scripts/generate_grasps.py --all`
  - Experimental runs: `python main.py --object_code_list "['banana']" --name test --n_contact 4 --batch_size 128 --n_iter 6000 --gpu "0"`.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, `snake_case` for functions/variables, `CamelCase` for classes.
- Keep script entry points in `scripts/` and reusable modules in `utils/`.
- Prefer explicit paths relative to the repo root (e.g., `data/meshdata/<object_code>/`).

## Testing Guidelines
- There is no formal unit test suite; validation is mostly via scripts.
- Smoke test visualization: `python grasp_generation/tests/visualize_result.py --object_code banana --num 0 --result_path ../data/experiments/<name>/results`.
- If you add tests, place lightweight checks under `grasp_generation/tests/` and document how to run them.

## Commit & Pull Request Guidelines
- Commit history favors short, imperative messages; optional prefix like `chore:` is used (e.g., `chore: add ...`).
- PRs should describe the change, include reproduction commands, and note any data dependencies or required external tools.
- If you update data formats or paths, include a brief migration note in the PR.

## Configuration Tips
- `data/` directories are typically symlinked to larger storage; avoid hardcoding absolute paths.
- GPU selection uses `CUDA_VISIBLE_DEVICES` or `--gpu` flags in scripts.
