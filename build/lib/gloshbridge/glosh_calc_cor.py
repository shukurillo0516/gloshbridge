import pandas as pd
import numpy as np

import hdbscan


class GLOSH:
    """Match Ref = True"""

    def __init__(self, data: np.ndarray, min_pts: int, min_clsize: int | None = None):
        self.data = data
        self.min_pts = min_pts
        self.min_clsize = min_clsize if min_clsize is not None else min_pts
        self.linkage_tree: pd.DataFrame | None = None
        self.glosh_scores = None

    def get_linkage_tree(self) -> pd.DataFrame:
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
        self.linkage_tree = clusterer.single_linkage_tree_.to_pandas()

        return self.linkage_tree

    def find_C(self, x: int) -> dict:
        """
        C_first: down to top traversal first cluster that joins x
        """
        C_first = None
        linkage_tree = self.get_linkage_tree()
        Cl = linkage_tree.loc[(linkage_tree["left_child"] == x) | (linkage_tree["right_child"] == x)].iloc[0].to_dict()

        while True:
            Cl_id = int(Cl["parent"])
            Cl_next_match = linkage_tree.loc[
                (linkage_tree["left_child"] == Cl_id)
                | (linkage_tree["right_child"] == Cl_id) & (linkage_tree["distance"] == Cl["distance"])
            ]
            if Cl_next_match.empty:
                Cl_next = {}
            else:
                Cl_next = Cl_next_match.iloc[0].to_dict()
            # Cl_next = linkage_tree.loc[linkage_tree["parent"] == Cl_id + 1].iloc[0].to_dict()

            if (
                Cl_next
                and (Cl_next["left_child"] == Cl_id or Cl_next["right_child"] == Cl_id)
                and Cl_next["distance"] == Cl["distance"]
            ):
                Cl = Cl_next
            else:
                if Cl["size"] >= self.min_clsize:
                    C_first = Cl
                    break
                else:
                    Cl = (
                        linkage_tree.loc[
                            (linkage_tree["left_child"] == Cl_id)
                            | (linkage_tree["right_child"] == Cl_id) & (linkage_tree["distance"] != Cl["distance"])
                        ]
                        .iloc[0]
                        .to_dict()
                    )
        return C_first

    def find_eps_min(self, C_x: dict):
        linkage_tree = self.get_linkage_tree()
        eps_min = float("inf")

        def _find_eps_min(cluster):
            nonlocal eps_min
            eps = cluster["distance"]
            cl_size = cluster["size"]

            # Update eps_min if this cluster qualifies
            if cl_size >= self.min_clsize and eps < eps_min:
                eps_min = eps

            # Traverse children if they exist in the tree
            for child_key in ("left_child", "right_child"):
                match = linkage_tree.loc[(linkage_tree["parent"] == cluster[child_key])]
                if not match.empty:
                    _find_eps_min(match.iloc[0].to_dict())

        _find_eps_min(C_x)
        return eps_min

    def find_eps(self, x):
        linkage_tree = self.get_linkage_tree()
        point = (
            linkage_tree.loc[(linkage_tree["left_child"] == x) | (linkage_tree["right_child"] == x)].iloc[0].to_dict()
        )
        return point["distance"]

    def _calc_glosh_score(self, x: int):
        if self.data.shape[0] < self.min_clsize:
            return 0
        C_x = self.find_C(x)
        eps_min = self.find_eps_min(C_x)
        eps = self.find_eps(x)

        if eps == 0:
            return 0

        score = 1 - eps_min / eps
        return max(0, score)

    def calc_glosh_scores(self):
        m, n = self.data.shape

        self.glosh_scores = np.zeros(m)
        for x, _ in enumerate(self.data):
            self.glosh_scores[x] = self._calc_glosh_score(x)

        return self.glosh_scores


import numpy as np
import pandas as pd
import hdbscan
from functools import lru_cache


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
