#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse.linalg as linalg
import krylov

from logging import getLogger, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)


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
        A = Jacobi(func, x0, fx)
        dx = _inv(A, -fx)
        x0 = x0 + dx
    raise RuntimeError("Not convergent (Newton)")


def hook_step(A, b, r, nu=0, maxiter=100, e=0.1):
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
    (numpy.array, float)
        argmin of xi, and nu

    """
    r2 = r * r
    I = np.matrix(np.identity(len(b), dtype=b.dtype))
    AA = A.T * A
    Ab = np.dot(A.T, b)
    for t in range(maxiter):
        B = np.array(np.linalg.inv(AA - nu * I))
        xi = np.dot(B, Ab)
        Psi = np.dot(xi, xi)
        logger.debug("Psi:{:e}".format(Psi))
        if abs(Psi - r2) < e * r2:
            value = np.dot(xi, np.dot(AA, xi) + Ab)
            print("value", value)
            return xi, nu
        dPsi = 2 * np.dot(xi, np.dot(B, xi))
        a = - Psi * Psi / dPsi
        b = - Psi / dPsi - nu
        nu = a / r2 - b
    raise RuntimeError("Not convergent (hook-step)")


def krylov_hook_step(A, b, r, **kwds):
    """
    optimal hook step with Krylov subspace method

    Parameters
    ----------
    A : LinearOperator
    b : numpy.array
    r : float
        trusted region

    Returns
    --------
    (numpy.array, float)
        argmin of xi, and nu

    """
    H, V = krylov.arnoldi(A, b)
    beta = np.zeros(H.shape[0])
    beta[0] = np.linalg.norm(b)
    xi, nu = hook_step(H, beta, r, **kwds)
    return np.dot(V, xi), nu


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
        beta = np.zeros(H.shape[0])
        beta[0] = np.linalg.norm(b)
        xi, _ = hook_step(H, beta, r)
        logger.debug("xi:" + str(xi))
        dx = np.dot(V, xi)
        logger.debug("dx:" + str(dx))
        x0 = x0 + dx
    raise RuntimeError("Not convergent (Newton-hook)")
