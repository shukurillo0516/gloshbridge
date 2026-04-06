import numpy as np
import pandas as pd
import hdbscan
from functools import lru_cache


# Must be tested, not tested yet
class FastGLOSH:
    def __init__(self, data: np.ndarray, min_pts: int, min_clsize: int | None = None):
        self.data = data
        self.min_pts = min_pts
        self.min_clsize = min_clsize if min_clsize is not None else min_pts
        self.linkage_tree = None

    def get_linkage_tree(self):
        if self.linkage_tree is not None:
            return self.linkage_tree

        clusterer = hdbscan.HDBSCAN(
            alpha=1.0,
            approx_min_span_tree=False,
            gen_min_span_tree=True,
            metric="euclidean",
            min_cluster_size=self.min_clsize,
            min_samples=self.min_pts,
            allow_single_cluster=False,
            match_reference_implementation=True,
        )
        clusterer.fit(self.data)
        tree_df = clusterer.single_linkage_tree_.to_pandas()

        # Convert to NumPy arrays for fast access
        self.linkage_tree = {
            "parent": tree_df["parent"].to_numpy(dtype=np.int64),
            "left": tree_df["left_child"].to_numpy(dtype=np.int64),
            "right": tree_df["right_child"].to_numpy(dtype=np.int64),
            "dist": tree_df["distance"].to_numpy(dtype=np.float64),
            "size": tree_df["size"].to_numpy(dtype=np.int64),
        }
        return self.linkage_tree

    @lru_cache(maxsize=None)
    def _find_C(self, x: int):
        tree = self.get_linkage_tree()
        left, right, parent, dist, size = (tree["left"], tree["right"], tree["parent"], tree["dist"], tree["size"])

        # find first occurrence where x is left or right
        mask = (left == x) | (right == x)
        if not mask.any():
            return None
        i = np.argmax(mask)
        while True:
            cl_id = parent[i]
            mask_next = (left == cl_id) | (right == cl_id)
            same_dist_mask = mask_next & (dist == dist[i])
            if not same_dist_mask.any():
                break
            i = np.argmax(same_dist_mask)
            if size[i] >= self.min_clsize:
                return i
        return i

    def _calc_glosh_score(self, x: int):
        tree = self.get_linkage_tree()
        left, right, parent, dist, size = (tree["left"], tree["right"], tree["parent"], tree["dist"], tree["size"])
        if self.data.shape[0] < self.min_clsize:
            return 0

        i = self._find_C(x)
        if i is None:
            return 0

        eps_min = np.min(dist[size >= self.min_clsize])
        eps = dist[(left == x) | (right == x)][0]

        if eps == 0:
            return 0
        return max(0, 1 - eps_min / eps)

    def calc_glosh_scores(self):
        m = self.data.shape[0]
        scores = np.zeros(m)
        for i in range(m):
            scores[i] = self._calc_glosh_score(i)
        return scores
