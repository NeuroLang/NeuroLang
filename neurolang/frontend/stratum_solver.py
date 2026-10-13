"""
Stratum solvers for the probabilistic frontend.

A stratum solver resolves the three strata of a program (deterministic,
probabilistic and post-probabilistic) on behalf of an engine. The default
:class:`StratumSolver` evaluates them through the engine's own stratum
methods; subclasses can change what is done with each stratum, e.g.
:class:`~neurolang.frontend.show_ra.ShowRAStratumSolver` prints the
relational algebra plans instead of evaluating them.
"""


class StratumSolver:
    """
    Evaluate each stratum of a program with the engine's solvers.

    Parameters
    ----------
    engine : NeurolangPDL
        engine whose program is solved.

    """

    def __init__(self, engine):
        """Store the engine whose strata are solved."""
        self.engine = engine

    def deterministic(self, det_idb):
        """Solve the deterministic stratum ``det_idb``."""
        return self.engine._solve_deterministic_stratum(det_idb)

    def probabilistic(self, solution, prob_idb):
        """Solve the probabilistic stratum ``prob_idb`` over ``solution``."""
        return self.engine._solve_probabilistic_stratum(solution, prob_idb)

    def postprobabilistic(self, solution, postprob_idb):
        """Solve the post-probabilistic stratum ``postprob_idb``."""
        return self.engine._solve_postprobabilistic_deterministic_stratum(
            solution, postprob_idb
        )
