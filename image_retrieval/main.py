"""
Image retrieval code
"""

from functools import reduce
import logging
from itertools import combinations
from os.path import exists
import pickle
from pathlib import Path
from multiprocessing import Manager, cpu_count, Pool

import cv2 as cv
import numpy as np
from rich.logging import RichHandler

from image_retrieval.helpers import (
    Invariants,
    nearest_points,
    calc_area,
    imread,
    log_all_methods,
)
from image_retrieval.config import IMG_DIR
from image_retrieval import global_vars

global_vars.initialize()


def parallelized_calc_invariant(feature_point, p, n, m, invariant):
    """
    this function needs to be at module level
    """
    npoints = nearest_points(feature_point, p, n)
    ret = []

    for mask in combinations(np.arange(n), m):
        r = ImageRetriever.calc_invariant(npoints[list(mask)], invariant)
        ret.append((p, r))

    return ret


def parallelized_query(hash_table, feature_point, ps, n, m, k, invariant, max_size):
    """
    parallelized query
    """
    ret = []

    for p in ps:
        npoints = nearest_points(feature_point, p, n)

        for mask in combinations(np.arange(n), m):
            # m points from n
            mpoints = npoints[list(mask)]
            for _ in range(m):
                # cyclic permutation of m points
                mpoints = np.roll(mpoints, 1, axis=0)

                r = ImageRetriever.calc_invariant(mpoints, invariant)
                hindex = ImageRetriever.calc_index(r, k, max_size)

                if hindex not in hash_table:
                    continue
                # voting
                for item in hash_table[hindex]:
                    # condition 1
                    if np.allclose(r, item[2]):
                        ret.append([item[0], tuple(p), tuple(item[1])])

    return ret


@log_all_methods(
    ignore=[
        "register",
        "calc_invariant",
        "calc_index",
        "calc_votes",
    ]
)
class ImageRetriever:
    def __init__(
        self,
        max_size=128 * 1e6,
        invariant=Invariants.AFFINE,
        n=7,
        m=6,
        k=25,
        parallel_count=cpu_count(),
    ):
        self.manager = Manager()
        self.hash_table = self.manager.dict()
        self.max_size = max_size
        self.invariant = invariant
        self.n = n
        self.m = m
        self.k = k
        self.parallel_count = parallel_count

    @staticmethod
    def calc_index(r, K, size):
        """
        calculates hash index
        """
        ans, k = 0, 1
        r = r.astype("int64") % size

        for i in range(r.shape[0]):
            ans = (ans + r[i] * k) % size
            k = k * K % size

        return ans

    @staticmethod
    def calculate_feature_point(img: np.ndarray):
        """
        Calculates features for the given image
        """
        img = img.copy()

        # step 1: adaptive thresholding of the input image
        img_bin = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10
        )

        # step 2: Gaussian filtering with square root of mode of areas of connected components
        labels = cv.connectedComponents(255 - img_bin)[1]
        areas = np.unique(labels, return_counts=True)[1]
        areas, cnt = np.unique(areas, return_counts=True)

        k = int(np.sqrt(areas[np.argmax(cnt)]))
        # make k odd
        k += k % 2 == 0

        img_blur = cv.GaussianBlur(img_bin, (k, k), 0)

        img_blur_bin = cv.adaptiveThreshold(
            img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10
        )

        labels = cv.connectedComponents(255 - img_blur_bin)[1]
        coordinates = np.argwhere(labels != -1)
        labels = labels.ravel()
        centroids = np.vstack([np.bincount(labels, weights=coordinates[:, 0]), np.bincount(labels, weights=coordinates[:, 1])])
        cnt = np.bincount(labels)
        cnt = np.where(cnt == 0, 1, cnt)
        
        return np.unique((centroids / cnt).astype('int64').T, axis=0)

        # contours = cv.findContours(
        #     255 - img_blur_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        # )[0]
        # centroids = []

        # for cnt in contours:
        #     M = cv.moments(cnt)
        #     cx, cy = 0, 0
        #     if M["m00"] != 0:
        #         cx = M["m01"] / M["m00"]
        #         cy = M["m10"] / M["m00"]
        #     centroids.append((cx, cy))

        # return np.unique(np.array(centroids, dtype=np.int64), axis=0)

    @staticmethod
    def calc_invariant(points, invariant):
        """
        calculates mCf points for affine or cross-ratio invariance
        """
        m, r = points.shape[0], []
        try:
            if invariant == Invariants.AFFINE:
                for mask in combinations(np.arange(m), 4):
                    p = points[list(mask)]
                    r.append(calc_area(p[0], p[2], p[3]) / calc_area(p[0], p[1], p[2]))
            elif invariant == Invariants.CROSS_RATIO:
                for mask in combinations(np.arange(m), 5):
                    p = points[list(mask)]
                    r.append(
                        calc_area(p[0], p[1], p[2])
                        * calc_area(p[0], p[3], p[4])
                        / (calc_area(p[0], p[1], p[3]) * calc_area(p[0], p[2], p[4]))
                    )
            return np.array(r)
        except:
            return np.array(r)

    def calculate_features(self, img, parallel=True):
        """
        calculate features for an image in the database
        """
        feature_point = ImageRetriever.calculate_feature_point(img)
        n, m = (
            min(self.n, feature_point.shape[0] - 1),
            min(self.m, feature_point.shape[0] - 1),
        )
        if n < 0 or m < 0:
            return []

        flatten = lambda x, y: x + y

        if parallel:
            pool = Pool(self.parallel_count)
            return reduce(
                flatten,
                pool.starmap(
                    parallelized_calc_invariant,
                    map(
                        lambda p: (feature_point, p, n, m, self.invariant),
                        feature_point,
                    ),
                ),
            )

        features = []
        for p in feature_point:
            features.append(
                parallelized_calc_invariant(feature_point, p, n, m, self.invariant)
            )
        return reduce(flatten, features)

    def register(self, doc_id, point_id, r):
        """
        register features to the hash table
        """
        hindex = ImageRetriever.calc_index(r, self.k, self.max_size)

        if hindex in self.hash_table:
            self.hash_table[hindex].append([doc_id, point_id, r])
        else:
            self.hash_table[hindex] = [[doc_id, point_id, r]]

        return hindex

    def save_hash_table(self, filename="hash_table.pickle"):
        """
        saves hash table to file
        """
        with open(filename, "wb") as f:
            pickle.dump(dict(self.hash_table), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_hash_table(self, filename="hash_table.pickle"):
        """
        loads hash table
        """
        if not exists(filename):
            files = list(Path(IMG_DIR).glob("*.png"))
            for idx, f in enumerate(files):
                feats = self.calculate_features(imread(f.name, mode=0))
                for feat in feats:
                    self.register(idx, *feat)
            self.save_hash_table(filename)
        else:
            with open(filename, "rb") as f:
                self.hash_table = self.manager.dict(pickle.load(f))

    @staticmethod
    def calc_votes(votes):
        """
        adds votes to the voting table which satisfies condition 2 and 3
        returns array of pairs of votes, doc_id sorted by votes in descending order
        """
        voting_table = {}

        for vote in votes:
            # condition 2 and 3
            if vote[0] not in voting_table:
                voting_table[vote[0]] = [set([vote[1]]), set([vote[2]])]
            elif (
                vote[1] not in voting_table[vote[0]][0]
                and vote[2] not in voting_table[vote[0]][1]
            ):
                voting_table[vote[0]][0].add(vote[1])
                voting_table[vote[0]][1].add(vote[2])

        # doc ids with votes
        doc_ids = []
        for key in voting_table:
            vote = len(voting_table[key][0])
            doc_ids.append((vote, key))
        doc_ids.sort(reverse=True)
        return np.array(doc_ids)

    def query(self, img, parallel=True):
        """
        query the db for document images similar to the one provided
        """
        feature_point = ImageRetriever.calculate_feature_point(img)
        n, m = (
            min(self.n, feature_point.shape[0] - 1),
            min(self.m, feature_point.shape[0] - 1),
        )
        if n < 0 or m < 0:
            return []

        flatten = lambda x, y: x + y

        if parallel:
            ps = np.array_split(feature_point, self.parallel_count)
            args = []
            for i in range(self.parallel_count):
                args.append(
                    (
                        self.hash_table,
                        feature_point,
                        ps[i],
                        n,
                        m,
                        self.k,
                        self.invariant,
                        self.max_size,
                    )
                )

            pool = self.manager.Pool(self.parallel_count)
            votes = reduce(
                flatten,
                pool.starmap(parallelized_query, args),
            )
        else:
            votes = []
            for p in feature_point:
                votes.append(
                    parallelized_query(
                        self.hash_table,
                        feature_point,
                        np.array([p]),
                        n,
                        m,
                        self.k,
                        self.invariant,
                        self.max_size,
                    )
                )
            votes = reduce(flatten, votes)

        return ImageRetriever.calc_votes(votes)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    ir = ImageRetriever()
    files = list(Path(IMG_DIR).glob("*.png"))

    ir.load_hash_table()
    # for idx, f in enumerate(files):
    #     feats = ir.calculate_features(imread(f.name, mode=0))

    #     for feat in feats:
    #         ir.register(idx, *feat)

    print(ir.query(imread(files[0].name, mode=0)))
