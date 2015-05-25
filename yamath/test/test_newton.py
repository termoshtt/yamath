# -*- coing: utf-8 -*-

from .. import newton

import numpy as np
from scipy.sparse import linalg as sclinalg
from unittest import TestCase


def f1(v):
    x = v[0]
    y = v[1]
    return np.array([(x - 1) * (x - 2) * (x - 3) * (x - 4), (y - 2) ** 3])


class TestNewton(TestCase):

    def test_newton(self):
        x0 = np.array([0, 0])
        x = newton.newton(f1, x0)
        np.testing.assert_array_almost_equal(f1(x), np.zeros_like(x), decimal=5)

    def test_hook_step(self):
        N = 5
        r = 0.1
        A = np.random.rand(N, N)
        I = np.identity(N)
        b = np.random.rand(N)
        xi, nu = newton.hook_step(A, b, r)
        np.testing.assert_almost_equal(np.linalg.norm(xi), r, decimal=1)
        B = A.T * A - nu * I
        np.testing.assert_array_almost_equal(np.dot(B, xi), np.dot(A.T, b))

    def test_krylov_hook_step(self):
        N = 5
        r = 0.1
        A = np.random.rand(N, N)
        b = np.random.rand(N)
        B = sclinalg.aslinearoperator(A)
        xi1, nu1 = newton.hook_step(A, b, r, e=1e-1)
        print(nu1)
        xi2, nu2 = newton.krylov_hook_step(B, b, r, e=1e-1)
        print(nu2)
        np.testing.assert_almost_equal(xi1, xi2, decimal=8)
