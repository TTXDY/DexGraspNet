import argparse
import csv
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm


def _load_size_map(path: str) -> Dict[str, list]:
    with open(path, "r") as f:
        data = json.load(f)
    return {k.lower(): v for k, v in data.items()}


def _load_shapenet_names(shapenet_root: str) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """
    Build mapping: (synsetId, modelId) -> (SynsetName, SubSynsetName)
    synsetId may have leading zeros; normalize by stripping leading zeros.
    """
    name_csv = os.path.join(shapenet_root, "all_name.csv")
    mapping: Dict[Tuple[str, str], Tuple[str, str]] = {}
    if not os.path.exists(name_csv):
        return mapping
    with open(name_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            synset = row.get("synsetId") or row.get("SynsetId") or row.get("synsetID")
            model = row.get("modelId") or row.get("modelID")
            syn_name = row.get("SynsetName") or ""
            sub_name = row.get("SubSynsetName") or ""
            if not synset or not model:
                continue
            synset_norm = str(synset).lstrip("0")
            mapping[(synset_norm, model)] = (syn_name, sub_name)
    return mapping


def _guess_category(synset: str, model: str,
                    shapenet_map: Dict[Tuple[str, str], Tuple[str, str]],
                    prefer_subsynset: bool) -> Optional[str]:
    synset_norm = synset.lstrip("0")
    key = (synset_norm, model)
    if key not in shapenet_map:
        return None
    syn_name, sub_name = shapenet_map[key]
    name = sub_name if prefer_subsynset and sub_name else syn_name
    if not name:
        return None
    return name.split(",")[0].strip().lower()


def _synset_label(synset: str, shapenet_map: Dict[Tuple[str, str], Tuple[str, str]]) -> str:
    for (s, _), (syn_name, _) in shapenet_map.items():
        if s == synset.lstrip("0") and syn_name:
            return syn_name.split(",")[0].strip().lower()
    return synset


def _match_size(name: str, size_map: Dict[str, list]) -> Optional[np.ndarray]:
    if name in size_map:
        return np.array(size_map[name], dtype=np.float32)
    for key, value in size_map.items():
        if key in name:
            return np.array(value, dtype=np.float32)
    return None


def _scale_mesh(mesh: trimesh.Trimesh, target_size: np.ndarray, anisotropic: bool) -> trimesh.Trimesh:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    if verts.size == 0:
        return mesh
    bounds = np.vstack([verts.min(axis=0), verts.max(axis=0)])
    size = bounds[1] - bounds[0]
    if np.any(size <= 0):
        return mesh
    if anisotropic:
        scale = target_size / size
        verts_scaled = (verts - bounds.mean(axis=0)) * scale + bounds.mean(axis=0)
    else:
        scale = target_size.max() / size.max()
        verts_scaled = verts * scale
    return trimesh.Trimesh(vertices=verts_scaled, faces=mesh.faces, process=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_root", type=str, required=True,
                        help="Path to ShapeNetCore.v2/ShapeNetCore.v2")
    parser.add_argument("--dst_root", type=str, required=True,
                        help="Output root for scaled ShapeNet models")
    parser.add_argument("--size_map", type=str, required=True,
                        help="JSON mapping from category name to target size [x,y,z] in meters")
    parser.add_argument("--prefer_subsynset", action="store_true",
                        help="Prefer SubSynsetName over SynsetName for category mapping")
    parser.add_argument("--anisotropic", action="store_true",
                        help="Scale per-axis to match target size (default: uniform scale)")
    args = parser.parse_args()

    size_map = _load_size_map(args.size_map)
    shapenet_map = _load_shapenet_names(args.shapenet_root)

    os.makedirs(args.dst_root, exist_ok=True)

    synsets = [d for d in os.listdir(args.shapenet_root)
               if os.path.isdir(os.path.join(args.shapenet_root, d))]
    skipped = 0
    synset_name_to_dir = {}
    for synset in tqdm(sorted(synsets), desc="synsets", mininterval=1.0):
        syn_dir = os.path.join(args.shapenet_root, synset)
        models = [d for d in os.listdir(syn_dir)
                  if os.path.isdir(os.path.join(syn_dir, d))]
        syn_label = _synset_label(synset, shapenet_map)
        # handle name collisions by suffix
        if syn_label in synset_name_to_dir and synset_name_to_dir[syn_label] != synset:
            syn_label = f"{syn_label}_{synset}"
        synset_name_to_dir[syn_label] = synset
        for model in tqdm(models, desc=f"models/{synset}", leave=False, mininterval=1.0):
            cat = _guess_category(synset, model, shapenet_map, args.prefer_subsynset)
            if cat is None:
                skipped += 1
                continue
            target = _match_size(cat, size_map)
            if target is None:
                skipped += 1
                continue
            src_mesh = os.path.join(syn_dir, model, "models", "model_normalized.ply")
            if not os.path.exists(src_mesh):
                skipped += 1
                continue
            mesh = trimesh.load(src_mesh, force="mesh", process=False)
            mesh_scaled = _scale_mesh(mesh, target, args.anisotropic)
            dst_dir = os.path.join(args.dst_root, syn_label, model, "models")
            os.makedirs(dst_dir, exist_ok=True)
            dst_mesh = os.path.join(dst_dir, "model_normalized.ply")
            mesh_scaled.export(dst_mesh)
    print(f"Done. Skipped {skipped} models without size mapping or missing mesh.")
