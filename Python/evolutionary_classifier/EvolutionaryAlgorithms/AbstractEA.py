import numpy as np

class AbstractEA(object):
    """"abstract" base class for an optimizer interface.

    """

    def __init__(self, xstart, **more_kwargs):
        """abstract method, ``xstart`` is a mandatory argument """
        raise NotImplementedError('method to be implemented in subclass')

    def ask(self):
        """abstract method, AKA get, deliver new candidate solution(s),
        a list of "vectors" """
        raise NotImplementedError('method to be implemented in subclass')

    def tell(self, solutions, function_values):
        """abstract method, AKA update, prepare for next iteration"""
        raise NotImplementedError('method to be implemented in subclass')

    def stop(self):
        """abstract method, return satisfied termination conditions in
        a dictionary like ``{'termination reason':value, ...}``,
        for example ``{'tolfun':1e-12}``, or the empty dictionary ``{}``.
        The implementation of `stop()` should prevent an infinite loop.

        """
        raise NotImplementedError('method to be implemented in subclass')

    def result(self):
        """abstract method, return ``(x, f(x), ...)``, that is the
        minimizer, its function value, ..."""
        raise NotImplementedError('method to be implemented in subclass')

    def disp(self, verbosity_modulo=1):
        """display of some iteration info"""
        print("method disp of " + str(type(self)) + " is not implemented")

    def optimize(self, objectivefct, iterations=None, min_iterations=1,
                 args=(), verb_disp=20, logger=None):
        """iterate at least ``min_iterations`` and at most ``iterations``
        using objective function ``objectivefct``.  Prints current information every ``verb_disp``,
        uses ``OptimDataLogger logger``, and returns the number of
        conducted iterations.

        Example
        =======
        ::

            import barecmaes2 as cma
            es = cma.CMAES(7 * [0.1], 0.5).optimize(cma.Fcts.rosenbrock)
            print(es.result()[0])

        """
        if iterations is not None and iterations < min_iterations:
            raise ValueError("iterations was smaller than min_iterations")
        if iterations == 0:
            raise ValueError("parameter iterations = 0")

        iteration = 0
        while not self.stop() or iteration < min_iterations:
            if iterations and iteration >= iterations:
                return self
            iteration += 1

            X = self.ask()  # deliver candidate solutions
            fitvals = np.apply_along_axis(objectivefct, 1, X, *args)

            self.tell(X, fitvals)  # all the work is done here

            if verb_disp:
                self.disp(verb_disp)

            logger.add(self) if logger else None

        logger.add(self, force=True) if logger else None
        if verb_disp:
            self.disp(1)
            print('termination by', self.stop())
            print('best f-value =', self.result()[1])
            print('solution =', self.result()[0])

        return self