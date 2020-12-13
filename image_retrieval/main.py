"""
Image retrieval code
"""

from functools import reduce
import logging
from itertools import combinations
from os.path import exists
from pathlib import Path
from multiprocessing import Manager, cpu_count, Pool

import sqlite3
import math
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


def parallelized_query(feature_point, ps, n, m, k, invariant, max_size, filename):
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

                with sqlite3.connect(filename) as conn:
                    cur = conn.cursor()
                    query = f"SELECT * FROM hash_table WHERE hindex={hindex};"
                    # voting
                    for row in cur.execute(query):
                        # condition 1
                        if np.allclose(r, np.array(row[4:])):
                            ret.append([row[1], tuple(p), (row[2], row[3])])

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
        invariant=Invariants.CROSS_RATIO,
        n=7,
        m=6,
        k=25,
        parallel_count=cpu_count(),
    ):
        self.max_size = max_size
        self.invariant = invariant
        self.n = n
        self.m = m
        self.k = k
        self.parallel_count = parallel_count
        self.conn = None
        self.cur = None
        self.connect_db()

    @staticmethod
    def nCr(n,r):
        f = math.factorial
        return f(n) // f(r) // f(n-r)

    def create_db(self, filename):
        with sqlite3.connect(filename) as conn:
            cur, points = conn.cursor(), 0
            if self.invariant == Invariants.AFFINE:
                points = ImageRetriever.nCr(self.m, 4)
            elif self.invariant == Invariants.CROSS_RATIO:
                points = ImageRetriever.nCr(self.m, 5)
            query = "CREATE TABLE hash_table (hindex INT, doc_id INT, px INT, py INT"
            for i in range(points):
                query += ", r" + str(i) + " FLOAT"
            query += ");"
            cur.execute(query)

    def connect_db(self, filename="hash_table.db"):
        if not exists(filename):
            self.create_db(filename)
            self.conn = sqlite3.connect(filename)
            self.cur = self.conn.cursor()
            self.preprocess()
        else:
            self.conn = sqlite3.connect(filename)
            self.cur = self.conn.cursor()

    def preprocess(self):
        """
        calculates feature points for each image
        """
        files = list(Path(IMG_DIR).glob("*.png"))
        for idx, f in enumerate(files):
            self.register_db(idx, self.calculate_features(imread(f.name, mode=0)))

    def register_db(self, doc_id, feats):
        """
        register features in db
        """
        if len(feats) == 0: return
        values = []
        for feat in feats:
            p, r = feat
            values.append((ImageRetriever.calc_index(r, self.k, self.max_size), doc_id, int(p[0]), int(p[1]), *r))
        query = "INSERT INTO hash_table VALUES (?,?,?,?"
        for _ in range(len(feats[0][1])):
            query += ",?"
        query += ");"
        self.cur.executemany(query, values)
        self.conn.commit()

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
    def calculate_feature_point(img: np.ndarray, k=5):
        """
        Calculates features for the given image
        """
        img = img.copy()

        img_bin = cv.adaptiveThreshold(
            img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10
        )

        # # step 2: Gaussian filtering with square root of mode of areas of connected components
        # labels = cv.connectedComponents(255 - img_bin)[1]
        # areas = np.unique(labels, return_counts=True)[1]
        # areas, cnt = np.unique(areas, return_counts=True)

        # k = int(np.sqrt(areas[np.argmax(cnt)]))
        # # make k odd
        # k += k % 2 == 0

        # img_bin = cv.GaussianBlur(img_bin, (k, k), 0)

        # img_bin = cv.adaptiveThreshold(
        #     img_bin, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10
        # )

        img_bin = 255 - cv.dilate(255 - img_bin, np.ones((k, k)))

        # # step 3: Calculating Centroids
        # labels = cv.connectedComponents(255 - img_bin)[1]
        # coordinates = np.argwhere(labels != -1)
        # labels = labels.ravel()
        # centroids = np.vstack([np.bincount(labels, weights=coordinates[:, 0]), np.bincount(labels, weights=coordinates[:, 1])])
        # cnt = np.bincount(labels)
        # cnt = np.where(cnt == 0, 1, cnt)
        
        # return img_bin, np.unique((centroids / cnt).astype('int64').T, axis=0)

        contours, _ = cv.findContours(
            255 - img_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        centroids = []
        for cnt in contours:
            M = cv.moments(cnt)
            cx, cy = 0, 0
            if M["m00"] != 0:
                cx = M["m01"] / M["m00"]
                cy = M["m10"] / M["m00"]
            centroids.append((cx, cy))

        return img_bin, np.unique(np.array(centroids, dtype=np.int64), axis=0)

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
        feature_point = ImageRetriever.calculate_feature_point(img)[1]
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

    def query(self, img, filename="hash_table.db", parallel=True):
        """
        query the db for document images similar to the one provided
        """
        feature_point = ImageRetriever.calculate_feature_point(img)[1]
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
                        feature_point,
                        ps[i],
                        n,
                        m,
                        self.k,
                        self.invariant,
                        self.max_size,
                        filename,
                    )
                )

            pool = Pool(self.parallel_count)
            votes = reduce(
                flatten,
                pool.starmap(parallelized_query, args),
            )
        else:
            votes = []
            for p in feature_point:
                votes.append(
                    parallelized_query(
                        feature_point,
                        np.array([p]),
                        n,
                        m,
                        self.k,
                        self.invariant,
                        self.max_size,
                        filename,
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

    print(ir.query(imread(files[0].name, mode=0)[500:1500, 200:1000]))
