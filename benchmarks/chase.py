from neurolang import expression_walker as ew
from neurolang import expressions
from neurolang.datalog.basic_representation import DatalogProgram
from neurolang.datalog.chase import (ChaseGeneral, ChaseMGUMixin, ChaseNaive,
                                     ChaseNamedRelationalAlgebraMixin,
                                     ChaseRelationalAlgebraPlusCeriMixin,
                                     ChaseSemiNaive)
from neurolang.datalog.expressions import Implication, TranslateToLogic

C_ = expressions.Constant
S_ = expressions.Symbol
Imp_ = Implication
Eb_ = expressions.ExpressionBlock

Q = S_('Q')
T = S_('T')
v = S_('v')
w = S_('w')
x = S_('x')
y = S_('y')
z = S_('z')


class Datalog(TranslateToLogic, DatalogProgram, ew.ExpressionBasicEvaluator):
    pass


class TimeChase:

    params = [
        [1000],
        [
            ChaseNaive,
            ChaseSemiNaive
        ],
        [
            ChaseMGUMixin,
            ChaseNamedRelationalAlgebraMixin,
            ChaseRelationalAlgebraPlusCeriMixin,
        ]
    ]

    param_names = [
        'triples in the DB',
        'chase strategy',
        'conjunctive query execution'
    ]

    def setup(self, n, chase_strategy, cq_compilation):
        from numpy import random

        rstate = random.RandomState(0)
        t = rstate.randint(
            0, max(n // 100, 2), size=(3, n)
        )

        dl = Datalog()
        dl.add_extensional_predicate_from_tuples(
            T,
            (tuple(row) for row in t)
        )

        class Chase(chase_strategy, cq_compilation, ChaseGeneral):
            pass

        self.dl = dl
        self.chase = Chase

    def time_selection(self, n, chase_strategy, cq_compilation):
        self.dl.push_scope()
        self.dl.walk(Eb_([
            Imp_(Q(x, y), T(C_(100), x, y))
        ]))
        self.chase(self.dl).build_chase_solution()
        self.dl.push_scope()

    def time_join(self, n, chase_strategy, cq_compilation):
        self.dl.push_scope()
        self.dl.walk(Eb_([
            Imp_(Q(x, y), T(x, z) & T(z, y))
        ]))
        self.chase(self.dl).build_chase_solution()
        self.dl.push_scope()

    def time_project(self, n, chase_strategy, cq_compilation):
        self.dl.push_scope()
        self.dl.walk(Eb_([
            Imp_(Q(x), T(x, y, z))
        ]))
        self.chase(self.dl).build_chase_solution()
        self.dl.push_scope()

    def time_recursion(self, n, chase_strategy, cq_compilation):
        self.dl.push_scope()
        self.dl.walk(Eb_([
            Imp_(Q(x, z), T(x, y, w) & Q(w, z)),
            Imp_(Q(x, z), T(x, y, z)),
        ]))
        self.chase(self.dl).build_chase_solution()
        self.dl.push_scope()
