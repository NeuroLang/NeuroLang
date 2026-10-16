from types import SimpleNamespace

from ._backend import BACKENDS, use_backend


def _datalog_namespace():
    """
    Import the NeuroLang classes and symbols used by the benchmarks.

    They are imported after the backend is selected (see ``use_backend``),
    so that the Datalog program stores its relations with that backend.
    """
    from neurolang import expression_walker as ew
    from neurolang import expressions
    from neurolang.datalog import chase
    from neurolang.datalog.basic_representation import DatalogProgram
    from neurolang.datalog.expressions import Implication, TranslateToLogic

    class Datalog(
        TranslateToLogic, DatalogProgram, ew.ExpressionBasicEvaluator
    ):
        pass

    S_ = expressions.Symbol
    return SimpleNamespace(
        chase=chase,
        Datalog=Datalog,
        C_=expressions.Constant,
        Imp_=Implication,
        Eb_=expressions.ExpressionBlock,
        **{name: S_(name) for name in ('Q', 'T', 'w', 'x', 'y', 'z')}
    )


class TimeChase:

    params = [
        [1000],
        [
            'ChaseNaive',
            'ChaseSemiNaive'
        ],
        [
            'ChaseMGUMixin',
            'ChaseNamedRelationalAlgebraMixin',
            'ChaseRelationalAlgebraPlusCeriMixin',
        ],
        BACKENDS,
    ]

    param_names = [
        'triples in the DB',
        'chase strategy',
        'conjunctive query execution',
        'backend',
    ]

    def setup(self, n, chase_strategy, cq_compilation, backend):
        from numpy import random

        use_backend(backend)
        self.ns = ns = _datalog_namespace()

        rstate = random.RandomState(0)
        t = rstate.randint(
            0, max(n // 100, 2), size=(3, n)
        )

        dl = ns.Datalog()
        dl.add_extensional_predicate_from_tuples(
            ns.T,
            (tuple(row) for row in t)
        )

        class Chase(
            getattr(ns.chase, chase_strategy),
            getattr(ns.chase, cq_compilation),
            ns.chase.ChaseGeneral,
        ):
            pass

        self.dl = dl
        self.chase = Chase

    def _solve(self, *rules):
        self.dl.push_scope()
        self.dl.walk(self.ns.Eb_(rules))
        self.chase(self.dl).build_chase_solution()
        self.dl.pop_scope()

    def time_selection(self, n, chase_strategy, cq_compilation, backend):
        ns = self.ns
        Q, T, x, y = ns.Q, ns.T, ns.x, ns.y
        self._solve(ns.Imp_(Q(x, y), T(ns.C_(100), x, y)))

    def time_join(self, n, chase_strategy, cq_compilation, backend):
        ns = self.ns
        Q, T, x, y, z = ns.Q, ns.T, ns.x, ns.y, ns.z
        self._solve(ns.Imp_(Q(x, y), T(x, z) & T(z, y)))

    def time_project(self, n, chase_strategy, cq_compilation, backend):
        ns = self.ns
        Q, T, x, y, z = ns.Q, ns.T, ns.x, ns.y, ns.z
        self._solve(ns.Imp_(Q(x), T(x, y, z)))

    def time_recursion(self, n, chase_strategy, cq_compilation, backend):
        ns = self.ns
        Q, T, w, x, y, z = ns.Q, ns.T, ns.w, ns.x, ns.y, ns.z
        self._solve(
            ns.Imp_(Q(x, z), T(x, y, w) & Q(w, z)),
            ns.Imp_(Q(x, z), T(x, y, z)),
        )
