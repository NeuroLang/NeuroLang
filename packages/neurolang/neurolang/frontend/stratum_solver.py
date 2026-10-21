"""
Stratum solvers for the probabilistic frontend.

A stratum solver resolves the three strata of a program (deterministic,
probabilistic and post-probabilistic) on behalf of an engine. The default
:class:`StratumSolver` evaluates them through the engine's own stratum
methods; subclasses can change what is done with each stratum, e.g.
:class:`~neurolang.frontend.show_ra.ShowRAStratumSolver` prints the
relational algebra plans instead of evaluating them.
"""
from typing import AbstractSet

from nl_expr import expressions as ir
from nl_core.datalog.expression_processing import extract_logic_atoms
from nl_core.datalog.wrapped_collections import WrappedRelationalAlgebraSet
from nl_core.relational_algebra import NamedRelationalAlgebraFrozenSet


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


def with_empty_relations(solution, prob_idb, program):
    """
    Return the relations of ``solution`` and the empty ones ``prob_idb`` uses.

    The chase leaves out of its solution the deterministic intensional
    predicates of ``program`` without tuples; the rules of ``prob_idb``
    need their relations, of their arity, to select constants from them.
    """
    idb = program.intensional_database()
    relations = dict(solution.items())
    heads = {rule.consequent.functor for rule in prob_idb.formulas}
    for rule in prob_idb.formulas:
        for atom in extract_logic_atoms(rule.antecedent):
            if atom.functor in idb and not (
                atom.functor in heads or atom.functor in relations
            ):
                relations[atom.functor] = _empty_relation(len(atom.args))
    return relations


def _empty_relation(arity):
    """Return an empty relation of ``arity`` columns."""
    columns = tuple(str(i) for i in range(arity))
    return ir.Constant[AbstractSet](
        WrappedRelationalAlgebraSet(
            NamedRelationalAlgebraFrozenSet(columns, []).to_unnamed()
        )
    )
