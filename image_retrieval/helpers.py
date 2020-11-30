"""
Common functions
"""

from enum import Enum
from pathlib import Path
from itertools import permutations, combinations
from functools import wraps
import inspect
import logging
import time

import cv2 as cv
import numpy as np

from image_retrieval import config, global_vars


class InvalidImage(Exception):
    """
    When asked to read an image but couldn't
    """


class Invariants(Enum):
    AFFINE = 1
    CROSS_RATIO = 2


def imread(name, _dir=config.IMG_DIR, mode=cv.IMREAD_COLOR):
    """
    Reads an image present in `_dir`
    """
    img_path = Path(_dir) / name
    img = cv.imread(str(img_path), mode)

    if img is None:
        raise InvalidImage("Couldn't read image from disk")

    return img


def imshow(img, window_name="", quit_char="q"):
    """
    displays an image
    waits until `quit_char` is pressed
    """
    cv.imshow(window_name, img)
    while True:
        if cv.waitKey() & 0xFF == ord(quit_char):
            cv.destroyAllWindows()
            break


def permute(n, m):
    """
    Returns a list(set) of tuples containing unique permutations of m True and n-m False
    """
    return set(permutations([x < m for x in range(n)]))


def calc_area(a, b, c, epsilon=1e-8):
    """
    calculates area given 3 2d points
    """
    mat = np.array([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]])
    return np.abs(np.linalg.det(mat) + epsilon) / 2


def cart2pol(points):
    """
    cartesian coordinates to polar coordinates
    """
    rho2 = np.sum(points ** 2, axis=1)
    phi = np.arctan2(points[:, 1], points[:, 0])
    return rho2, phi


def nearest_points(points, point, n):
    """
    n nearest points to a point in clockwise order
    """
    rho2, phi = cart2pol(points - point)
    idx = np.argsort(rho2)[1 : n + 1]
    # clockwise n nearest points
    return points[idx][np.argsort(phi[idx])]


def fmt_print(string, thresh_len=70):
    """
    Formats stuff for printing
    ensures that len(output) is around thresh_len
    """
    string = str(string)
    string = string.replace("\n", " ")

    if len(string) > thresh_len:
        half_thresh_len = thresh_len // 2
        string = string[:half_thresh_len] + " ... " + string[-half_thresh_len:]

    return " ".join(string.split())


def stringify_call_params(*args, **kwargs) -> str:
    """
    Returns a string generated from call params
    """
    str_args = list(map(str, args))
    str_kwargs = []
    for k, v in kwargs.items():
        str_kwargs.append(f"{k}={v}")

    return ", ".join(str_args + str_kwargs)


def log_call(f, log_entry=True, log_exit=True):
    """
    Wrapper for logging every function call
    """

    @wraps(f)
    def _log_call(*args, **kwargs):
        """
        add a try-catch block to func call
        """
        log = logging.getLogger()

        indent_offset = "-" * 4 * global_vars.logging_indent

        global_vars.logging_indent += 1

        with np.printoptions(edgeitems=1, precision=3):
            if log_entry:
                log.debug(
                    "%s -> %s.%s(%s)",
                    indent_offset,
                    f.__module__,
                    f.__qualname__,
                    fmt_print(stringify_call_params(*args, **kwargs),),
                )

            st = time.time()
            ret = f(*args, **kwargs)
            en = time.time()

            global_vars.logging_indent -= 1

            if log_exit:
                log.debug(
                    "%s <- [%.3f] %s.%s(%s)",
                    indent_offset,
                    en - st,
                    f.__module__,
                    f.__qualname__,
                    fmt_print(ret),
                )

        return ret

    return _log_call


# taken from https://stackoverflow.com/q/6307761
def log_all_methods(log_entry=True, log_exit=True, ignore=[]):
    """
    A decorator which logs all methods in a class
    """

    def decorate(cls):
        for attr in cls.__dict__:
            _func = getattr(cls, attr)
            if callable(_func) and attr not in ignore:
                setattr(
                    cls, attr, log_call(_func, log_entry, log_exit),
                )
        return cls

    return decorate


def log_all_in_module(module, log_entry=True, log_exit=True):
    """
    Adds the logging decorator to all methods in class
    """
    try:
        all_attrs = module.__all__
    except AttributeError:
        all_attrs = dir(module)

    for attr in all_attrs:
        _func = getattr(module, attr)
        if inspect.isclass(_func):
            setattr(module, attr, log_all_methods(log_entry, log_exit)(_func))
        elif callable(_func):
            setattr(module, attr, log_call(_func, log_entry, log_exit))
