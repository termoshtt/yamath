#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as linalg

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


def f1(v):
    x = v[0]
    y = v[1]
    return np.array([(x - 1) * (x - 2) * (x - 3) * (x - 4), (y - 2) ** 3])


def Jacobi(func, x0, alpha=1e-7, fx=None):
    if fx is None:
        fx = func(x0)

    def wrap(v):
        norm = np.linalg.norm(v)
        r = alpha / norm
        return (func(x0 + r * v) - fx) / r

    return linalg.LinearOperator((len(x0), len(x0)), matvec=wrap, dtype=x0.dtype)


def _inv(A, b):
    x, res = linalg.gmres(A, b)
    if res:
        logger.warning("Iteration of GMRES does not convergent, res={:d}".format(res))
        raise RuntimeError("Not convergent (GMRES)")
    return x


def newton(func, x0, ftol=1e-5, maxiter=100):
    for t in range(maxiter):
        fx = func(x0)
        res = np.linalg.norm(fx)
        logger.debug('count:{:d}\tresidual:{:e}'.format(t, res))
        if res <= ftol:
            return x0
        A = Jacobi(func, x0)
        dx = _inv(A, -fx)
        x0 += dx
    raise RuntimeError("Not convergent (Newton)")


def newton_hook(func, x0, r=1e-2, ftol=1e-5, maxiter=100):
    for t in range(maxiter):
        fx = func(x0)
        res = np.linalg.norm(fx)
        logger.debug('count:{:d}\tresidual:{:e}'.format(t, res))
        if res <= ftol:
            return x0
        A = Jacobi(func, x0, fx=fx)
        dx = _inv(A, -fx)
        dx_norm = np.linalg.norm(dx)
        logger.debug('dx_norm:{:e}'.format(dx_norm))
        if dx_norm < r:
            logger.info('in Trusted region')
            x0 += dx
            continue
        logger.info('hook step')
        x0 += dx  # TODO
    raise RuntimeError("Not convergent (Newton-hook)")


def hook_step(A, b, r, nu=0):
    r2 = r * r
    I = np.matrix(np.identity(len(b), dtype=b.dtype))
    AA = A.T * A
    Ab = A.T * b
    while True:
        B = np.linalg.inv(AA - nu * I)
        xi = B * Ab
        Psi = float(xi.T * xi)
        logger.debug("Psi:{:e}".format(Psi))
        if abs(Psi - r2) < 0.1 * r2:
            return xi
        dPsi = 2 * float(xi.T * B * xi)
        a = - Psi * Psi / dPsi
        b = - Psi / dPsi - nu
        nu = a / r2 - b

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.addHandler(handler)

    b = np.matrix([1, 0], dtype=np.float64).T
    A = np.matrix([[1, 0], [2, 6]], dtype=np.float64)
    xi = hook_step(A, b, 0.2)
    print(xi)
    print(np.linalg.norm(xi))
