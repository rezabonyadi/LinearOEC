'''
This is an efficient implementation of CMAES. The propoved method here is up to 5 times faster than the original
implementation in standard cma package 2.5.


Written by M.R.Bonyadi (rezabny@gmail.com)
'''

import sys
from random import normalvariate as random_normalvariate
import numpy as np
from random import normalvariate as random_normalvariate
from math import log
from EvolutionaryAlgorithms.AbstractEA import AbstractEA


class CMAES(AbstractEA):
    def __init__(self, xstart, sigma,  # mandatory
                 max_eval='1e3*N**2',
                 ftarget=None,
                 popsize="4 + int(3 * log(N))",
                 randn=random_normalvariate):
        """Initialize` CMAES` object instance, the first two arguments are
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
        self.sigma = sigma
        self.ftarget = ftarget  # stop if fitness < ftarget
        self.max_eval = eval(str(max_eval))  # eval a string
        self.randn = randn

        # Strategy parameter setting: Selection
        self.lam = eval(str(popsize))  # population size, offspring number
        self.mu = int(self.lam / 2)  # number of parents/points for recombination
        self.weights = [np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)]  # recombination weights
        self.weights = [w / sum(self.weights) for w in self.weights]  # normalize recombination weights array
        self.mueff = sum(self.weights) ** 2 / sum(w ** 2 for w in self.weights)  # variance-effectiveness of sum w_i x_i

        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)  # time constant for cumulation for C
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)  # t-const for cumulation for sigma control
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)  # learning rate for rank-one update of C
        self.cmu = min([1 - self.c1,
                        2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff)])  # and for rank-mu update
        self.damps = 2 * self.mueff / self.lam + 0.3 + self.cs  # damping for sigma, usually close to 1
        self.chiN = np.sqrt(self.dim) * (1 - 1. / (4. * self.dim) +
                                         1. / (21. * self.dim ** 2))
        # Initialize dynamic (internal) state variables
        self.pc, self.ps = np.zeros(N), np.zeros(N)  # evolution paths for C,sigma
        self.B = np.eye(N)  # B defines the coordinate system
        self.D = N * [1]  # diagonal D defines the scaling
        self.C = np.eye(N)  # covariance matrix
        self.invsqrtC = np.eye(N)  # C^-1/2
        self.eigeneval = 0  # tracking the update of B and D
        self.counteval = 0
        self.fitvals = []  # for bookkeeping output and termination

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
            if max(self.D) > 1e7 * min(self.D):
                res['condition'] = 1e7
            if len(self.fitvals) > 1 \
                    and self.fitvals[-1] - self.fitvals[0] < 1e-12:
                res['tolfun'] = 1e-12
            if self.sigma * max(self.D) < 1e-11:
                # remark: max(D) >= max(diag(C))**0.5
                res['tolx'] = 1e-11
        return res

    def ask(self):
        """return a list of lambda candidate solutions according to
        m + sig * Normal(0,C) = m + sig * B * D * Normal(0,I)"""

        # Eigendecomposition: first update B, D and invsqrtC from C
        # postpone in case to achieve O(N**2)
        if self.counteval - self.eigeneval > \
                self.lam / (self.c1 + self.cmu) / self.dim / 10:
            self.eigeneval = self.counteval
            self.D, self.B = np.linalg.eigh(self.C)  # eigen decomposition, B==normalized eigenvectors, O(N**3)
            self.D = [d ** 0.5 for d in self.D]  # D contains standard deviations now
            self.invsqrtC = np.dot((self.B / self.D), self.B.T)

        # lam vectors x_k = m  +          sigma * B * D * randn_k(N)

        arz = np.random.standard_normal((self.lam, self.dim))
        arz = self.xmean + self.sigma * np.dot(arz, np.transpose(self.B))
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
        iN = range(N)
        xold = self.xmean

        # Sort by fitness and compute weighted mean into xmean
        arindex = np.argsort(fitvals)
        self.fitvals = fitvals[arindex]
        # self.fitvals, arindex = np.sort(fitvals), np.argsort(fitvals)  # min
        arx = [arx[arindex[k]] for k in range(self.mu)]  # sorted arx
        del fitvals, arindex  # to prevent misuse
        # self.fitvals is kept for termination and display only
        self.best.update([arx[0]], [self.fitvals[0]], self.counteval)

        # xmean = [x_1=best, x_2, ..., x_mu] * weights
        self.xmean = self.dot(arx[0:self.mu], self.weights, t=True)
        # recombination, new mean value
        # == [sum(self.weights[k] * arx[k][i] for k in range(self.mu))
        #                                     for i in iN]

        # Cumulation: update evolution paths
        y = np.subtract(self.xmean, xold)
        z = np.dot(self.invsqrtC, y) # == C**(-1/2) * (xnew - xold)

        c = (self.cs * (2 - self.cs) * self.mueff) ** 0.5 / self.sigma
        for i in iN:
            self.ps[i] -= self.cs * self.ps[i]  # exponential decay on ps
            self.ps[i] += c * z[i]
        hsig = (sum(x ** 2 for x in self.ps)
                / (1 - (1 - self.cs) ** (2 * self.counteval / self.lam)) / N
                < 2 + 4. / (N + 1))
        c = (self.cc * (2 - self.cc) * self.mueff) ** 0.5 / self.sigma
        for i in iN:
            self.pc[i] -= self.cc * self.pc[i]  # exponential decay on pc
            self.pc[i] += c * hsig * y[i]

        artmp = np.subtract(arx, xold)
        self.C = np.multiply(1 - self.c1 - self.cmu + (1 - hsig) *
                             self.c1 * self.cc * (2 - self.cc), self.C) \
                 + self.c1 * np.outer(self.pc, self.pc) \
                 + self.cmu * np.dot((self.weights * np.transpose(artmp)), artmp) \
                 / self.sigma ** 2

        # Adapt step-size sigma with factor <= exp(0.6) \approx 1.82
        self.sigma *= np.exp(min(0.6, (self.cs / self.damps) *
                              (sum(x ** 2 for x in self.ps) / N - 1) / 2))

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
            max_std = max([self.C[i][i] for i in range(len(self.C))]) ** 0.5
            print(repr(self.counteval).rjust(5) + ': ' +
                  ' %6.1f %8.1e  ' % (max(self.D) / min(self.D),
                                      self.sigma * max_std) +
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

