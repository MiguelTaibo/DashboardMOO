import numpy as np
from scipy import spatial
from functools import reduce
import tensorflow as tf

def filter_(pts, pt):
    weakly_worse   = (pts >= pt).all(axis=-1)
    strictly_worse = (pts > pt).any(axis=-1)
    return pts[~(weakly_worse & strictly_worse)]

def get_pareto_undominated_by(pts1, pts2=None):
    if pts2 is None:
        pts2 = pts1
    return reduce(filter_, pts2, pts1)


def get_pareto_frontier(pts):
    pareto_groups = []
    while pts.shape[0]:
        if pts.shape[0] < 10:
            pareto_groups.append(get_pareto_undominated_by(pts))
            break

        hull_vertices = spatial.ConvexHull(pts).vertices
        hull_pts = pts[hull_vertices]
        nonhull_mask = np.ones(pts.shape[0], dtype=bool)
        nonhull_mask[hull_vertices] = False
        pts = pts[nonhull_mask]
        pareto   = get_pareto_undominated_by(hull_pts)
        pareto_groups.append(pareto)
        pts = get_pareto_undominated_by(pts, pareto)

    return np.vstack(pareto_groups)

def filterdict(list, key, value):
    return [el for el in list if el[key]==value]


