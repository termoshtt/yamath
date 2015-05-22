#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as linalg
import krylov

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


def hook_step(A, b, r, nu=0):
    """
    optimal hook step based on trusted region approach

    Parameters
    ----------
    A : numpy.array
        square matrix
    b : numpy.array
    r : float
        trusted region

    Returns
    --------
    numpy.array
        argmin of xi

    """
    r2 = r * r
    I = np.matrix(np.identity(len(b), dtype=b.dtype))
    AA = A.T * A
    Ab = np.dot(A.T, b)
    while True:
        B = np.array(np.linalg.inv(AA - nu * I))
        xi = np.dot(B, Ab)
        Psi = np.dot(xi, xi)
        logger.debug("Psi:{:e}".format(Psi))
        if abs(Psi - r2) < 0.1 * r2:
            return xi
        dPsi = 2 * np.dot(xi, np.dot(B, xi))
        a = - Psi * Psi / dPsi
        b = - Psi / dPsi - nu
        nu = a / r2 - b


def newton_hook(func, x0, r=1e-2, ftol=1e-5, maxiter=100):
    for t in range(maxiter):
        logger.debug("x0:" + str(x0))
        fx = func(x0)
        res = np.linalg.norm(fx)
        logger.debug('count:{:d}\tresidual:{:e}'.format(t, res))
        if res <= ftol:
            return x0
        A = Jacobi(func, x0, fx=fx)
        b = -fx
        dx = _inv(A, b)
        dx_norm = np.linalg.norm(dx)
        logger.debug('dx_norm:{:e}'.format(dx_norm))
        if dx_norm < r:
            logger.info('in Trusted region')
            x0 = x0 + dx
            continue
        logger.info('hook step')
        H, V = krylov.arnoldi(A, b)
        beta = np.array([np.linalg.norm(b)])
        beta.resize(H.shape[0])
        xi = hook_step(H, beta, r)
        logger.debug("xi:" + str(xi))
        dx = np.dot(V, xi)
        logger.debug("dx:" + str(dx))
        x0 = x0 + dx
    raise RuntimeError("Not convergent (Newton-hook)")

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.addHandler(handler)

    x0 = np.array([0, 0])
    newton_hook(f1, x0, r=0.1)
