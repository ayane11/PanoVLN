"""SimpleVLN evaluation semantics: published locations + Euclidean fastdtw.

This does not replace Pano's existing geodesic/densified-path measure. Callers
must record actual simulator positions and select the published GT by episode.
"""
import math

import numpy as np
from fastdtw import fastdtw


def _points(values, name):
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f'{name} must contain nonempty finite 3D positions')
    return points.tolist()


def reference_ndtw(positions, reference_locations, *, success_distance=3.):
    """Match SimpleVLN NDTW, including exact adjacent rollout deduplication."""
    if not math.isfinite(success_distance) or success_distance <= 0:
        raise ValueError('Success distance must be positive and finite')
    locations = []
    for point in _points(positions, 'positions'):
        if not locations or point != locations[-1]:
            locations.append(point)
    reference = _points(reference_locations, 'reference_locations')
    # Keep GT duplicates and its original sampling density. No interpolation,
    # shortest-path replacement, epsilon deduplication or geodesic distance.
    distance = fastdtw(locations, reference,
                      dist=lambda a, b: float(np.linalg.norm(np.asarray(a)-np.asarray(b))))[0]
    return float(np.exp(-float(distance)/(len(reference)*success_distance)))


def reference_path_metrics(positions, reference_locations, *, success, success_distance=3.):
    if success not in (0, 1):
        raise ValueError('Expected binary navigation success')
    ndtw = reference_ndtw(positions, reference_locations, success_distance=success_distance)
    return {'ndtw': ndtw, 'sdtw': float(success)*ndtw}
