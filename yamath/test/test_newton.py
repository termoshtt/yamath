# -*- coing: utf-8 -*-

from .. import newton

import numpy as np
from scipy.sparse import linalg as sclinalg
from unittest import TestCase


def f1(v):
    x = v[0]
    y = v[1]
    return np.array([(x - 1) * (x - 2) * (x - 3) * (x - 4), (y - 2) ** 3])


def f2(v):
    x = v[0]
    y = v[1]
    return np.array([x * x + 1, y * y + 2])


class TestNewton(TestCase):

    def test_newton(self):
        """ Simple test for Newton method using polynominal function """
        x0 = np.array([0, 0])
        x = newton.newton(f1, x0)
        np.testing.assert_array_almost_equal(f1(x), np.zeros_like(x), decimal=5)

    def test_newton_noconvergent(self):
        """ non-convergent case for Newton method """
        x0 = np.array([3, 8])
        with self.assertRaises(RuntimeError):
            newton.newton(f2, x0)

    def test_hook_step(self):
        """ fuzzy test of hook step """
        N = 5
        r = 0.1
        A = np.random.rand(N, N)
        b = np.random.rand(N)
        xi, nu = newton.hook_step(A, b, r)
        np.testing.assert_almost_equal(np.linalg.norm(xi), r, decimal=1)

    def test_krylov_hook_step(self):
        """ fuzzy test of Krylov-hook step """
        N = 5
        r = 0.1
        A = np.random.rand(N, N)
        b = np.random.rand(N)
        B = sclinalg.aslinearoperator(A)
        xi1, nu1 = newton.hook_step(A, b, r, e=1e-1)
        xi2, nu2 = newton.krylov_hook_step(B, b, r, e=1e-1)
        np.testing.assert_almost_equal(xi1, xi2, decimal=1)

    def test_newton_krylov_hook(self):
        """ Simple test for Newton-Krylov-hook method using polynominal function """
        x0 = np.array([0, 0])
        x = newton.newton_krylov_hook(f1, x0, r=0.2)
        np.testing.assert_array_almost_equal(f1(x), np.zeros_like(x), decimal=5)

    def test_newton_krylov_hook_nozero(self):
        """ non-convergent case (no-zero point) for Newton-Krylov-hook """
        x0 = np.array([3, 8])
        with self.assertRaises(RuntimeError):
            newton.newton_krylov_hook(f2, x0, r=0.2)
