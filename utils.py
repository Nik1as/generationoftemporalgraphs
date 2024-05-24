import networkit as nk

import itertools
import os
import sys
import math

from typing import Iterable


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def batch(iterable: Iterable, size: int):
    it = iter(iterable)
    while item := list(itertools.islice(it, size)):
        yield item


def powerlaw(x: int, gamma: float, c: float):
    return c * x**(-gamma)


def file_name(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def number_of_digits(x: int):
    return int(math.log10(x)) + 1


def node_count_iterator(start, end) -> Iterable[int]:
    n = start

    while n <= end:
        yield n

        n += 10**(number_of_digits(n) - 1)


def chung_lu_powerlaw(n: int, gamma: float) -> nk.Graph:
    degree_seq = nk.generators.PowerlawDegreeSequence(1, n - 1, -gamma).run().getDegreeSequence(n)
    return nk.generators.ChungLuGenerator(degree_seq).generate()
