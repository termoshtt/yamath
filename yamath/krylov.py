#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as linalg

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


class Arnoldi(object):

    def __init__(self, A, b):
        self.A = A
        b_norm = np.linalg.norm(b)
        self.basis = [b / b_norm]
        self.H = []

    def iterate(self):
        v = self.basis[-1]
        if len(self.H) > len(v):
            raise RuntimeError("Over iterate")
        u = self.A * v
        weight = []
        for b in self.basis:
            w = np.dot(u, b)
            weight.append(w)
            u -= w * b
        u_norm = np.linalg.norm(u)
        weight.append(u_norm)
        self.H.append(np.array(weight))
        if u_norm > 1e-10:
            b = u / u_norm
            self.basis.append(b)
            return b
        return None

    def get_basis(self):
        N = len(self.basis[0])
        resized = [np.resize(b, (1, N)) for b in self.basis]
        return np.concatenate(resized, axis=0)

    def get_projected_matrix(self):
        N = len(self.basis[0])
        resized = []
        for i, h in enumerate(self.H):
            tmp = np.resize(h, (N, 1))
            for j in range(i + 2, N):
                tmp[j, 0] = 0
            resized.append(tmp)
        return np.concatenate(resized, axis=1)


if __name__ == '__main__':
    rand = np.random.rand(3, 3)

    def matvec(x):
        return np.dot(rand, x)

    A = linalg.LinearOperator((3, 3), matvec=matvec, dtype=np.float64)
    b = np.array([1, 0, 0], dtype=np.float64)
    O = Arnoldi(A, b)
    while O.iterate() is not None:
        pass
    print(O.get_projected_matrix())
    print(O.get_basis())
