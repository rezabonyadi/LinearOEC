'''
This is a python implementation of the linear Optimal-margin Evolutionary Classifier (OEC). OEC only supports linear
kernels and uses derivative-free optimization to optimize the 0-1 loss. Hence, it is the most rebust classifier to
outlayers and provides closest possible solutions to the optimal 0-1 loss solution within a practical timeframe. It is
up to 20 times slower than state-of-the-art methods (SVM, LDA, etc.), but provides significantly better solutions with
much better generalization ability. Current implementation only supports CMAES and ES for optimization purposes.



Written by M.R.Bonyadi (rezabny@gmail.com)
'''

import numpy as np
from EvolutionaryAlgorithms.CMAES import CMAES
from EvolutionaryAlgorithms.ES import ES


class LinearOEC:
    optimizer = 'cmaes'
    regularization = 0
    initialization = None
    weights = []
    threshold = 0
    iterations = 0
    m = 0
    n = 0

    def __init__(self, n, optimizer='cmaes', regularization=0, initialization=None, iterations=None):
        self.initialization = initialization
        self.optimizer = optimizer
        self.regularization = regularization
        self.n = n
        if iterations is None:
            self.iterations = np.ceil(np.log(self.n+1)*150)
        else:
            self.iterations = iterations

    def fit(self, data, classes):
        # Classes is an array of 0 and 1, data is m by n, m is number of instances, n is the number of variables.

        [self.weights, self.threshold] = self.__calculate_weights__(data, classes)

    def predict_proba(self, data):
        # Returns the distance to the hyperplane, negatives are class zero while positives are class one.

        d = np.dot(data, self.weights) - self.threshold
        return -d

    def predict(self, data):
        # Returns the class labels for the given data. The result is a 0/1 array.

        classes = np.sign(self.predict_proba(data))
        classes += 1
        classes /= 2
        return classes

    def __calculate_weights__(self, data, classes):
        m = data.shape[0]
        self.priors1 = classes.sum()
        self.priors0 = m - self.priors1

        if self.optimizer == 'cmaes':
            if self.initialization is None:
                es = CMAES(self.n * [0], 1)
                es.optimize(self.__objective__, iterations=self.iterations,
                            args=(data, classes, self.priors0, self.priors1), verb_disp=0)

            else:
                es = CMAES(self.initialization, .1)
                es.optimize(self.__objective__, iterations=self.iterations,
                            args=(data, classes, self.priors0, self.priors1), verb_disp=0)
        else:
            if self.initialization is None:
                es = ES(self.n * [0], 1.0)
                es.optimize(self.__objective__, iterations=self.iterations,
                            args=(data, classes, self.priors0, self.priors1), verb_disp=0)

            else:
                es = ES(self.n * [0], 1)
                es.optimize(self.__objective__, iterations=self.iterations,
                            args=(data, classes, self.priors0, self.priors1), verb_disp=0)

        [thr, coef, perf, marg]= self.__optimal_discrimination__(np.dot(data, es.best.x), classes, self.priors0, self.priors1)
        return es.best.x * coef, thr

    def __objective__(self, w, data, c, t0, t1):
        d = np.dot(data, w)
        [thr, coef, f, marg] = self.__optimal_discrimination__(d, c, t0, t1)
        if (f==0):
            f = f-marg
        if self.regularization > 0.0:
            f = f + self.regularization * sum(abs(w))

        return f

    @staticmethod
    def __optimal_discrimination__(d, c, t0, t1):
        d_i = d.argsort()
        ds = d[d_i]
        cs = c[d_i]
        n1ls = np.cumsum(cs)
        n0ls = np.cumsum(-(cs-1.0))
        l0 = np.divide(n0ls, t0)
        l1 = np.divide(n1ls, t1)
        acc1 = l0 + (1.0 - l1)
        acc2 = 2.0 - acc1 # l1 + (1 - l0)
        ind1 = np.argmax(acc1)
        ind2 = np.argmax(acc2) - 1
        a1 = acc1[ind1]
        a2 = acc2[ind2]
        if a1 > a2:
            thr = -((ds[ind1] + ds[ind1 + 1]) / 2.0)
            coef = -1
            perf = a1
            marg = abs(ds[ind1] - ds[ind1 + 1])
        else:
            thr = ((ds[ind2] + ds[ind2 + 1]) / 2.0)
            coef = 1
            perf = a2
            marg = abs(ds[ind2] - ds[ind2 + 1])
        perf /= 2.0
        perf = 1 - perf
        return thr, coef, perf, marg


