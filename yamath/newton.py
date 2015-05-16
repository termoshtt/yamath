#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as linalg


def f1(v):
    x = v[0]
    y = v[1]
    return np.array([(x-1)*(x-2)*(x-3)*(x-4), (y-2)**3])


def Jacobi(func, x0, alpha=1e-7):
    def wrap(v):
        norm = np.linalg.norm(v)
        r = alpha / norm
        return (func(x0 + r*v) - func(x0)) / r
    return linalg.LinearOperator((len(x0), len(x0)), matvec=wrap, dtype=x0.dtype)


def newton(func, x0, ftol=1e-5, maxiter=100):
    for t in range(maxiter):
        fx = func(x0)
        res = np.linalg.norm(fx)
        print(res)
        if res <= ftol:
            return x0
        A = Jacobi(func, x0)
        dx, res = linalg.gmres(A, fx)
        x0 -= dx
    raise RuntimeError("Not convergent")


if __name__ == '__main__':
    x0 = np.array([0, 0], dtype=np.float64)
    x = newton(f1, x0)
    print(x)
    print(f1(x))
