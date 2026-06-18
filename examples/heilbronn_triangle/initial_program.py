# EVOLVE-BLOCK-START
import numpy as np
import itertools
import math


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points with a deterministic max-min geometric search.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    height = math.sqrt(3.0) / 2.0
    margin = 0.035

    def min_triangle_area(points: np.ndarray) -> float:
        best = float("inf")
        for i, j, k in itertools.combinations(range(len(points)), 3):
            a, b, c = points[i], points[j], points[k]
            area = 0.5 * abs(
                (b[0] - a[0]) * (c[1] - a[1])
                - (c[0] - a[0]) * (b[1] - a[1])
            )
            if area < best:
                best = area
        return best

    def project_to_triangle(points: np.ndarray) -> np.ndarray:
        v = points[:, 1] / height
        u = points[:, 0] - 0.5 * v
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        outside = u + v > 1.0
        total = u[outside] + v[outside]
        u[outside] /= total
        v[outside] /= total
        return np.column_stack((u + 0.5 * v, height * v))

    idx = np.arange(1, 1601, dtype=float)
    u = (idx * 0.7548776662466927) % 1.0
    v = (idx * 0.5698402909980532) % 1.0
    reflected = u + v > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    u = margin + (1.0 - 3.0 * margin) * u
    v = margin + (1.0 - 3.0 * margin) * v
    candidates = np.column_stack((u + 0.5 * v, height * v))

    selected = [
        np.array([0.5, height / 3.0]),
        np.array([1.5 * margin, height * margin]),
        np.array([1.0 - 1.5 * margin, height * margin]),
        np.array([0.5, height * (1.0 - 2.0 * margin)]),
    ]

    while len(selected) < 11:
        best_idx = 0
        best_score = -1.0
        for candidate_idx, candidate in enumerate(candidates):
            if any(np.linalg.norm(candidate - point) < 1e-12 for point in selected):
                continue
            score = min_triangle_area(np.array(selected + [candidate]))
            if score > best_score:
                best_score = score
                best_idx = candidate_idx
        selected.append(candidates[best_idx])

    points = np.array(selected, dtype=float)
    best_points = points.copy()
    best_score = min_triangle_area(points)

    rng = np.random.default_rng(7)
    step = 0.055
    for iteration in range(8000):
        candidate = points.copy()
        move_idx = int(rng.integers(0, 11))
        candidate[move_idx] += rng.normal(0.0, step, size=2)
        candidate = project_to_triangle(candidate)
        score = min_triangle_area(candidate)
        current = min_triangle_area(points)
        if score > best_score:
            best_score = score
            best_points = candidate.copy()
            points = candidate
        elif rng.random() < math.exp((score - current) / max(1e-8, step * 0.01)):
            points = candidate
        if (iteration + 1) % 1000 == 0:
            step *= 0.65

    return best_points


# EVOLVE-BLOCK-END
