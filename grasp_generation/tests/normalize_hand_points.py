#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime


def _normalize_points(points):
    if points is None:
        return []
    if not isinstance(points, list):
        raise ValueError(f"points must be a list, got {type(points)}")
    if len(points) == 0:
        return []
    is_flat = isinstance(points[0], (float, int))
    is_nested = isinstance(points[0], list)
    if is_flat:
        if len(points) % 3 != 0:
            raise ValueError(f"flat list length must be multiple of 3, got {len(points)}")
        return [points[i:i + 3] for i in range(0, len(points), 3)]
    if is_nested:
        for i, row in enumerate(points):
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"bad row {i}: {row}")
        return points
    raise ValueError(f"unsupported points format: {type(points[0])}")


def normalize_file(path: Path):
    data = json.loads(path.read_text())
    out = {}
    for k, v in data.items():
        out[k] = _normalize_points(v)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.stem + f"_backup_{ts}.json")
    backup.write_text(json.dumps(data, indent=2))
    path.write_text(json.dumps(out, indent=2))
    print(f"normalized {path} (backup {backup})")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "mjcf_dexhand021"
    normalize_file(base / "penetration_points.json")
    normalize_file(base / "contact_points.json")
