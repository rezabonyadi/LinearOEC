'''
This is an efficient implementation of Evolutionary Strategy (ES).


Written by M.R.Bonyadi (rezabny@gmail.com)
'''

import sys
from random import normalvariate as random_normalvariate
import numpy as np
from random import normalvariate as random_normalvariate
from math import log
from EvolutionaryAlgorithms.AbstractEA import AbstractEA


class ES(AbstractEA):
    def __init__(self, xstart, sigma,  # mandatory
                 max_eval='1e3*N**2',
                 ftarget=None,
                 popsize="4 + int(3 * log(N))",
                 randn=random_normalvariate):
        """Initialize` ES` object instance, the first two arguments are
        mandatory.

        Parameters
        ----------
            `xstart`
                ``list`` of numbers (like ``[3, 2, 1.2]``), initial
                solution vector
            `sigma`
                ``float``, initial step-size (standard deviation in each
                coordinate)
            `max_eval`
                ``int`` or ``str``, maximal number of function
                evaluations, a string is evaluated with ``N`` being the
                search space dimension
            `ftarget`
                `float`, target function value
            `randn`
                normal random number generator, by default
                ``random.normalvariate``

        """
        # process input parameters
        N = len(xstart)  # number of objective variables/problem dimension
        self.dim = N
        self.xmean = xstart[:]  # initial point, distribution mean, a copy
        self.ftarget = ftarget  # stop if fitness < ftarget
        self.max_eval = eval(str(max_eval))  # eval a string
        self.randn = randn

        # Strategy parameter setting: Selection
        self.lam = eval(str(popsize))  # population size, offspring number
        self.sigma = np.ones(self.lam) * sigma
        self.mu = int(self.lam / 2)  # number of parents/points for recombination

        self.counteval = 0
        self.fitvals = []  # for bookkeeping output and termination
        self.tau = (1 / (np.sqrt(2 * self.dim)))

        self.best = BestSolution()

    def stop(self):
        """return satisfied termination conditions in a dictionary like
        {'termination reason':value, ...}, for example {'tolfun':1e-12},
        or the empty dict {}"""
        res = {}
        if self.counteval > 0:
            if self.counteval >= self.max_eval:
                res['evals'] = self.max_eval
            if self.ftarget is not None and len(self.fitvals) > 0 \
                    and self.fitvals[0] <= self.ftarget:
                res['ftarget'] = self.ftarget
            if len(self.fitvals) > 1 \
                    and self.fitvals[-1] - self.fitvals[0] < 1e-12:
                res['tolfun'] = 1e-12
        return res

    def ask(self):
        """return a list of lambda candidate solutions according to
        m + sig * Normal(0,C) = m + sig * B * D * Normal(0,I)"""

        arz = np.random.standard_normal((self.lam, self.dim))

        arz = self.xmean + np.dot(np.diag(self.sigma), arz)

        arz = arz.T / np.sqrt(np.einsum('...i,...i', arz, arz))
        res = arz.T

        return res

    def tell(self, arx, fitvals):
        """update the evolution paths and the distribution parameters m,
        sigma, and C within CMA-ES.

        Parameters
        ----------
            `arx`
                a list of solutions, presumably from `ask()`
            `fitvals`
                the corresponding objective function values

        """

        # bookkeeping, preparation
        self.counteval += len(fitvals)  # slightly artificial to do here
        N = self.dim  # convenience short cuts

        # Sort by fitness and compute weighted mean into xmean
        arindex = np.argsort(fitvals)
        self.fitvals = fitvals[arindex]
        self.sigma = self.sigma[arindex]

        # self.fitvals, arindex = np.sort(fitvals), np.argsort(fitvals)  # min
        arx = [arx[arindex[k]] for k in range(self.mu)]  # sorted arx
        del fitvals, arindex  # to prevent misuse
        # self.fitvals is kept for termination and display only
        self.best.update([arx[0]], [self.fitvals[0]], self.counteval)

        recsigma = np.mean(self.sigma[0:self.mu])
        self.sigma = recsigma * (np.exp(self.tau * np.random.standard_normal(self.lam)))
        self.xmean = np.mean(arx, axis=0)

    def result(self):
        """return (xbest, f(xbest), evaluations_xbest, evaluations,
        iterations, xmean)

        """
        return self.best.get() + (self.counteval,
                                  int(self.counteval / self.lam),
                                  self.xmean)

    def disp(self, verb_modulo=1):
        """display some iteration info"""
        iteration = self.counteval / self.lam

        if iteration == 1 or iteration % (10 * verb_modulo) == 0:
            print('evals: ax-ratio max(std)   f-value')
        if iteration <= 2 or iteration % verb_modulo == 0:
            print(repr(self.counteval).rjust(5) + ': ' +
                  str(self.fitvals[0]))

            sys.stdout.flush()

        return None

    @staticmethod
    def dot(A, b, t=False):
        """ usual dot product of "matrix" A with "vector" b,
        where A[i] is the i-th row of A. With t=True, A transposed is used.

        :rtype : "vector" (list of float)
        """
        n = len(b)
        if not t:
            m = len(A)  # number of rows, like printed by pprint
            v = m * [0]
            for i in range(m):
                v[i] = sum(b[j] * A[i][j] for j in range(n))
        else:
            m = len(A[0])  # number of columns
            v = m * [0]
            for i in range(m):
                v[i] = sum(b[j] * A[j][i] for j in range(n))
        return v
    # -----------------------------------------------


class BestSolution(object):
    """container to keep track of the best solution seen"""

    def __init__(self, x=None, f=None, evals=None):
        """take `x`, `f`, and `evals` to initialize the best solution.
        The better solutions have smaller `f`-values. """
        self.x, self.f, self.evals = x, f, evals

    def update(self, arx, arf, evals=None):
        """initialize the best solution with `x`, `f`, and `evals`.
        Better solutions have smaller `f`-values."""
        if self.f is None or min(arf) < self.f:
            i = arf.index(min(arf))
            self.x, self.f = arx[i], arf[i]
            self.evals = None if not evals else evals - len(arf) + i + 1
        return self

    def get(self):
        """return ``(x, f, evals)`` """
        return self.x, self.f, self.evals