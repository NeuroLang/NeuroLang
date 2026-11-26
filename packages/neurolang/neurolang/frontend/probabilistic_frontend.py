r"""
Probabilistic Frontend
======================
Complements QueryBuilderDatalog class with probabilistic capabilities
1- add extensional probabilistic facts and choices
2- solve probabilistic queries
"""
import collections
import functools
import typing
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
)
from types import MappingProxyType
from uuid import uuid1

import pandas as pd
from nl_expr.type_system import get_args, replace_type_variable_fix_python36_37

from nl_expr import expressions as ir
from nl_core.datalog.aggregation import (
    BuiltinAggregationMixin,
    DatalogWithAggregationMixin,
    TranslateToLogicWithAggregation,
)
from nl_core.datalog.chase import Chase
from nl_core.datalog.constraints_representation import (
    DatalogConstraintsProgram,
    RightImplication,
    iter_implications,
)
from nl_core.datalog.exceptions import (
    InvalidMagicSetError,
    NewRecursionInMagicSetsRewriteError,
    NoConstantPredicateFoundError,
)
from nl_expr.exceptions import SymbolNotFoundError
from nl_core.datalog.expression_processing import (
    EqualitySymbolLeftHandSideNormaliseMixin,
    InlineEqualityConstantsMixin,
    extract_logic_atoms,
    is_aggregation_rule,
    reachable_code,
)
from nl_core.datalog.magic_sets import magic_rewrite
from nl_core.datalog.pretty_printer import DatalogPrettyPrinter
from nl_core.datalog.negation import DatalogProgramNegationMixin
from nl_core.datalog.ontologies_parser import OntologyParser
from nl_core.datalog.ontologies_rewriter import (
    OntologyRewriter,
    chase_rules,
    constraints_that_can_hold,
    projecting_rules,
    valued_rules,
)

from nl_expr.exceptions import (
    NeuroLangException,
    UnsupportedQueryError,
    UnsupportedSolverError,
    UnsupportedProgramError,
)
from nl_expr.expression_walker import (
    ExpressionBasicEvaluator,
    ReplaceExpressionWalker,
    TypedSymbolTableMixin,
)
from nl_expr.expressions import Constant
from nl_expr.logic import Conjunction, Implication, Union, Symbol
from nl_expr.logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables,
)
from nl_core.probabilistic import (
    dalvi_suciu_lift,
    small_dichotomy_theorem_based_solver,
)
from nl_core.probabilistic.cplogic.program import CPLogicMixin
from nl_core.probabilistic.expressions import PROB, Condition
from .resampling import (
    ResampledProgram,
    relation_sums,
    resampling_relation,
    rewrite_resample_markers,
)
from nl_core.probabilistic.expression_processing import (
    is_probabilistic_predicate_symbol,
    is_within_language_prob_query,
)
from nl_core.probabilistic.magic_sets_processing import (
    probabilistic_postprocess_magic_rules,
)
from nl_core.probabilistic.query_resolution import (
    QueryBasedProbFactToDetRule,
    compute_probabilistic_solution,
)
from nl_core.probabilistic.stratification import (
    get_list_of_intensional_rules,
    stratify_program,
)
from nl_core.probabilistic.weighted_model_counting import (
    solve_marg_query as wmc_solve_marg_query,
)
from nl_core.probabilistic.weighted_model_counting import (
    solve_succ_query as wmc_solve_succ_query,
)
from ..region_solver import RegionSolver
from nl_core.relational_algebra import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraColumnStr,
)
from nl_core.datalog.wrapped_collections import (
    WrappedRelationalAlgebraFrozenSet,
)
from ..commands import CommandsMixin
from nl_core.datalog.basic_representation import UnionOfConjunctiveQueries
from . import query_resolution_expressions as fe
from .datalog.sugar import (
    TranslateProbabilisticQueryMixin,
    TranslateQueryBasedProbabilisticFactMixin,
)
from .datalog.sugar.spatial import (
    TranslateEuclideanDistanceBoundMatrixMixin,
    TranslateRegionDestroy,
)
from .datalog.syntax_preprocessing import (
    ProbFol2DatalogMixin,
    rule_to_datalog,
)
from .type_resolution import ProbabilisticTypeResolutionMixin
from .datalog.squall import (
    CheckPredicateArities,
    ResolveInvertedFunctionApplicationMixin,
    StripDimensionTypePredicatesMixin,
)
from .frontend_extensions import NumpyFunctionsMixin
from .query_resolution_datalog import QueryBuilderDatalog
from .show_ra import (
    ShowRAStratumSolver,
    generated_query_symbol,
    show_ra_options,
)
from .stratum_solver import StratumSolver, with_empty_relations


def _symbol_names(symbols):
    """Return the names of ``symbols``."""
    return tuple(symbol.name for symbol in symbols)


def _constraints_of(frontend):
    """The constraints of the program of ``frontend``, the object of its
    symbol table, or None: the rewritings depend on them."""
    return frontend.program_ir.symbol_table.get(Symbol("__constraints__"))


class LoadedOntologies(NamedTuple):
    """What the ontologies loaded into a program define (see
    :meth:`NeurolangPDL.load_ontology`)."""

    #: the names of the predicates they define
    predicates: FrozenSet[str] = frozenset()
    #: the name of the connector symbol of the labels of their classes
    connector: Optional[str] = None
    #: the predicate of rdf:type
    type_predicate: Optional[str] = None
    #: the predicate of each class, by its IRI
    class_names: Mapping[str, str] = MappingProxyType({})
    #: the (individual, class IRI) of the rdf:type facts of the files
    stated_types: FrozenSet[Tuple[str, str]] = frozenset()


# In the symbol table: what the ontologies loaded, by scope.
LOADED_ONTOLOGIES = Symbol("__loaded_ontologies__")


def instance_lru_cache(key_fn, maxsize=128, valid_while=None):
    """LRU cache decorator for instance methods with non-hashable arguments.

    key_fn(*args, **kwargs) must return a hashable cache key.
    The cache is stored per-instance to avoid cross-instance sharing.
    Call wrapper.cache_clear(instance) to invalidate. With
    ``valid_while``, the cache is also invalidated when
    ``valid_while(instance)`` is no longer the object it was.
    """
    def decorator(method):
        cache_attr = f'_lru_cache_{method.__name__}'

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            stamp = None if valid_while is None else valid_while(self)
            cached = getattr(self, cache_attr, None)
            if cached is None or cached[0] is not stamp:
                cached = (stamp, collections.OrderedDict())
                setattr(self, cache_attr, cached)
            cache = cached[1]
            key = key_fn(*args, **kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
            result = method(self, *args, **kwargs)
            cache[key] = result
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        def cache_clear(instance):
            setattr(instance, cache_attr, None)

        wrapper.cache_clear = cache_clear
        return wrapper
    return decorator


class RegionFrontendCPLogicSolver(
    EqualitySymbolLeftHandSideNormaliseMixin,
    InlineEqualityConstantsMixin,
    TranslateProbabilisticQueryMixin,
    ResolveInvertedFunctionApplicationMixin,
    StripDimensionTypePredicatesMixin,
    TranslateToLogicWithAggregation,
    TranslateQueryBasedProbabilisticFactMixin,
    TranslateEuclideanDistanceBoundMatrixMixin,
    QueryBasedProbFactToDetRule,
    ProbFol2DatalogMixin,
    ProbabilisticTypeResolutionMixin,
    TranslateRegionDestroy,
    RegionSolver,
    CommandsMixin,
    NumpyFunctionsMixin,
    CPLogicMixin,
    DatalogWithAggregationMixin,
    BuiltinAggregationMixin,
    DatalogProgramNegationMixin,
    DatalogConstraintsProgram,
    TypedSymbolTableMixin,
    ExpressionBasicEvaluator,
):
    pass


class NeurolangPDL(QueryBuilderDatalog):
    """
    Probabilistic Datalog frontend.

    Complements :class:`QueryBuilderDatalog` with probabilistic
    capabilities:

    1. add extensional probabilistic facts and choices;
    2. solve probabilistic queries.
    """

    def __init__(
        self,
        chase_class: Type[Chase] = Chase,
        probabilistic_solvers: Tuple[Callable] = (
            small_dichotomy_theorem_based_solver.solve_succ_query,
            dalvi_suciu_lift.solve_succ_query,
            wmc_solve_succ_query,
        ),
        probabilistic_marg_solvers: Tuple[Callable] = (
            small_dichotomy_theorem_based_solver.solve_marg_query,
            dalvi_suciu_lift.solve_marg_query,
            wmc_solve_marg_query,
        ),
        check_qbased_pfact_tuple_unicity=False,
        solver_class: Optional[Type] = None,
    ) -> "NeurolangPDL":
        """
        Query builder with probabilistic capabilities

        Parameters
        ----------
        chase_class : Type[Chase], optional
            used to compute deterministic solutions, by default Chase
        probabilistic_solvers : Tuple[Callable], optional
            used to compute probabilistic solutions,
            the order of the elements indicates the priority
            of usage of the solver.
            by default (lifted_solve_succ_query, wmc_solve_succ_query)

        probabilistic_marg_solvers : Tuple[Callable], optional
            used to compute probabilistic solutions,
            by default (lifted_solve_marg_query, wmc_solve_marg_query)

        solver_class : Type, optional
            Custom solver class to use instead of RegionFrontendCPLogicSolver.
            Allows injecting custom mixins (e.g. with dimension_type_predicate_names
            tailored from engine YAML config).


        Returns
        -------
        NeurolangPDL
            see description
        """
        if solver_class is not None:
            solver = solver_class()
        else:
            solver = RegionFrontendCPLogicSolver()
        super().__init__(
            solver, chase_class=chase_class
        )
        if len(probabilistic_solvers) != len(probabilistic_marg_solvers):
            raise ValueError(
                "probabilistic solve and marg "
                "solvers should be the same length"
            )
        self.probabilistic_solvers = probabilistic_solvers
        self.probabilistic_marg_solvers = probabilistic_marg_solvers
        self.current_program_rewritten = None
        self._bridging = False
        self.check_qbased_pfact_tuple_unicity = (
            check_qbased_pfact_tuple_unicity
        )
        self.resamplings = {}
        self._resampled_versions = {}

    def load_ontology(
        self,
        paths: typing.Union[str, List[str]],
        connector_symbol_name=None,
        format=None,
        individuals=False,
        names="local",
        prefixes=None,
    ) -> None:
        """
        Loads and parses ontology stored at the specified paths, and
        store them into attributes

        Parameters
        ----------
        paths : typing.Union[str, List[str]]
            where the ontology files are stored
        connector_symbol_name : str
            name to be used in the connector_symbol.
            if None, the name is randomly generated
        format : str or list of str
            rdflib format of the files (``xml``, ``turtle``, ``nt``…),
            guessed from their extensions when None.
        individuals : bool
            also load every triple as a fact: ``s p o`` is ``p(s, o)``
            and ``s rdf:type C`` is also ``C(s)``, IRIs being the
            strings of their full IRI. The subclasses of the ontology
            apply to these facts: ``Super(x)`` holds for an instance of
            a subclass.
        names : str
            ``local`` names classes and properties by their local name
            (``Dean``, ``rdf-schema:label``); ``prefixed`` by the
            prefixes the files declare (``pv:name``, ``rdfs:label``)
            and the ``prefixes``, else by their full IRI.
        prefixes : dict, optional
            more prefixes for ``names="prefixed"``, from prefix to
            namespace, such as ``{"cogat":
            "http://www.cognitiveatlas.org/ontology/cogat.owl#"}`` for a
            file that declares none for it.

        Returns
        -------
        Expression
            the new connector_symbol, to be used in
            the query program.
        """
        # The ontologies loaded one after the other add up: they share
        # the connector symbol, and their constraints, rules and facts
        # are added to those of the ontologies loaded before.
        loaded = self._loaded_ontologies
        connector = connector_symbol_name or loaded.connector or str(uuid1())
        self._set_loaded_ontologies(connector=connector)
        loaded = loaded.predicates
        onto = OntologyParser(
            paths, connector_symbol=self.connector_symbol.expression,
            load_format=format, individuals=individuals, names=names,
            prefixes=prefixes,
        )
        constraints, est_knowledge, entity_rules = onto.parse_ontology()
        self._set_loaded_ontologies(predicates=loaded | set(constraints) | {
            s.name for s in (*est_knowledge, *entity_rules)
        })
        self.program_ir.add_constraints(constraints)
        for name, expressions in constraints.items():
            symbol = Symbol(name=name).cast(UnionOfConjunctiveQueries)
            before = self.program_ir.symbol_table.get(symbol, Union(()))
            self.program_ir.symbol_table[symbol] = Union(
                before.formulas + tuple(expressions)
            )
        for symbol, expressions in est_knowledge.items():
            rows = {
                tuple(arg.value for arg in exp.args) for exp in expressions
            }
            self.program_ir.add_extensional_predicate_from_tuples(
                symbol, self._loaded_rows(symbol.name, loaded) | rows
            )
        self.program_ir.add_existential_rules(onto.existential_rules)
        for symbol, expressions in entity_rules.items():
            for e in expressions:
                self.program_ir.walk(e)
        self._set_loaded_ontologies(
            class_names=MappingProxyType(
                {**self._class_names, **onto.class_names()}
            )
        )
        if individuals:
            self._set_loaded_ontologies(type_predicate=onto.type_name())
            classes = set(self._class_names.values())
            for name, rows in onto.parse_individuals().items():
                if name in classes:
                    # the instances of a class are its rdf:type facts,
                    # through _type_bridges, as those the program states
                    continue
                if name == self._type_predicate:
                    self._add_stated_types(rows)
                else:
                    self.add_tuple_set(
                        self._loaded_rows(name, loaded) | set(rows),
                        name=name,
                    )
                self._set_loaded_ontologies(
                    predicates=self.ontology_predicates | {name}
                )
        return self.connector_symbol

    def _add_stated_types(self, rows):
        """The ``rdf:type`` facts of the files, in a relation of their
        own that a rule of ``rdf:type`` reads: the rules of a program can
        state more types, ``rdf:type(e, t) :- …``."""
        stated = Symbol(f"{self._type_predicate}@files")
        before = self._loaded_rows(stated.name, {stated.name})
        self.add_tuple_set(before | set(rows), name=stated.name)
        self._set_loaded_ontologies(stated_types=self._stated_types | {
            (x, c) for x, c in rows if c in self._class_names
        })
        if not before:
            x, c = Symbol.fresh(), Symbol.fresh()
            self.program_ir.walk(Union((Implication(
                Symbol(self._type_predicate)(x, c), stated(x, c)
            ),)))

    # What the ontologies loaded, in the symbol table: it follows its
    # scopes, as the constraints, rules and facts of the ontologies do.

    @property
    def _loaded_ontologies(self):
        loaded = self.program_ir.symbol_table.get(LOADED_ONTOLOGIES)
        return LoadedOntologies() if loaded is None else loaded.value

    def _set_loaded_ontologies(self, **changes):
        self.program_ir.symbol_table[LOADED_ONTOLOGIES] = Constant(
            self._loaded_ontologies._replace(**changes),
            auto_infer_type=False, verify_type=False,
        )

    @property
    def ontology_predicates(self):
        """The names of the predicates the loaded ontologies define."""
        return self._loaded_ontologies.predicates

    @property
    def connector_symbol(self):
        """The symbol of the labels of the classes of the ontologies,
        ``connector(entity, label)``, or None if none was loaded."""
        name = self._loaded_ontologies.connector
        return None if name is None else self.new_symbol(name=name)

    @property
    def _type_predicate(self):
        """The predicate of rdf:type (see _type_bridges)."""
        return self._loaded_ontologies.type_predicate

    @property
    def _class_names(self):
        """The predicate of each class of the ontologies, by its IRI."""
        return self._loaded_ontologies.class_names

    @property
    def _stated_types(self):
        """The (individual, class IRI) of the rdf:type facts of the
        files."""
        return self._loaded_ontologies.stated_types

    def _loaded_rows(self, name, loaded):
        """The rows of the relation ``name``, if one of the ``loaded``
        predicates of the ontologies, else none."""
        relation = self.program_ir.symbol_table.get(Symbol(name))
        if name not in loaded or not isinstance(relation, Constant):
            return set()
        return set(relation.value.itervalues())

    def _closed_world_rows(self, body, variables):
        """The rows of ``variables`` in ``body``, by the rules and facts of
        the program alone: the ontologies' constraints do not apply, and
        none of their predicates may be reached."""
        constrained = {
            symbol for symbol in self.program_ir.constraints()._symbols
        }
        with self.scope:
            fresh = Symbol.fresh()
            self.program_ir.walk(Union((Implication(fresh(*variables), body),)))
            rule = self.program_ir.symbol_table[fresh].formulas[0]
            rules = reachable_code(rule, self.program_ir)
            reached = {
                atom.functor for r in rules.formulas
                for atom in extract_logic_atoms(r.antecedent)
            }
            if reached & constrained:
                return None
            solution = self.chase_class(
                self.program_ir, rules=rules
            ).build_chase_solution()
            relation = solution.get(fresh)
            if relation is None:
                return set()
            return {
                tuple(row)
                for row in relation.value.unwrap()
            }

    def _type_bridges(self):
        """
        The rules that make the instances of the classes of the
        ontologies from ``rdf:type``, RDF's meaning of it: ``C(x) :-
        rdf:type@files(x, <C>)`` for the types the files state, and, for a
        rule ``rdf:type(x, t) :- body`` of the program, ``C(x) :- body``
        with ``t`` the IRI of ``C``.  Only for the classes of which they
        give an instance: the rewriting drops the rewritings over the
        others.  Each class takes its instances from the data that type
        them, of whatever type their values are.
        """
        if (
            self._type_predicate is None
            or "__constraints__" not in self.symbol_table
        ):
            return ()
        x = Symbol.fresh()
        stated = Symbol(f"{self._type_predicate}@files")
        bridges = [
            Implication(
                Symbol(self._class_names[iri])(x),
                stated(x, Constant[str](iri)),
            )
            for iri in sorted({c for _, c in self._stated_types})
        ]
        rules = self.program_ir.symbol_table.get(
            Symbol(self._type_predicate)
        )
        for rule in getattr(rules, "formulas", ()):
            if stated in rule.antecedent._symbols:
                continue
            instance, iri_term = rule.consequent.args
            rows = self._closed_world_rows(
                rule.antecedent, tuple(
                    a for a in (instance, iri_term) if isinstance(a, Symbol)
                ),
            )
            if rows is None:
                raise NeuroLangException(
                    f"{self._type_predicate} is defined from a predicate the "
                    "ontology constrains: the types of the program must "
                    "come from its data"
                )
            if isinstance(iri_term, Symbol):
                iris = {row[-1] for row in rows}
            else:
                iris = {iri_term.value} if rows else set()
            for iri in sorted(i for i in iris if i in self._class_names):
                bridges.append(Implication(
                    Symbol(self._class_names[iri])(instance),
                    ReplaceExpressionWalker(
                        {iri_term: Constant[str](iri)}
                    ).walk(rule.antecedent) if isinstance(iri_term, Symbol)
                    else rule.antecedent,
                ))
        return tuple(bridges)

    def _typed_rules(self, rules):
        """The ``rules`` of a program, those with an ``rdf:type`` atom in
        their body translated to Datalog and with the classes they name
        (:meth:`_resolve_types`), with the rules they need."""
        if self._type_predicate is None:
            return list(rules)
        typed = []
        for rule in rules:
            if not isinstance(rule, Implication) or self._type_predicate not in {
                getattr(atom.functor, "name", None)
                for atom in extract_logic_atoms(rule.antecedent)
                if hasattr(atom, "functor")
            }:
                typed.append(rule)
                continue
            translated, needed = rule_to_datalog(rule, self.symbol_table)
            resolved, more = self._resolve_types(translated)
            typed += [resolved] + needed + more
        return typed

    def _resolve_types(self, rule):
        """
        The rule ``rule`` of a query with ``rdf:type(v, k)``, ``k`` fixed
        by atoms of the query without ``v`` and with no other variable, as
        ``cogat_term(k, 'response inhibition')``, as the rule of the class
        they name: ``C(v)``, which the ontology's constraints apply to,
        an unnamed ``v`` included.  Rules defining the query's head, one
        for each class when they name several, and the rules they need.
        """
        if self._type_predicate is None:
            return rule, []
        formulas = list(getattr(rule.antecedent, "formulas", (rule.antecedent,)))
        for atom in formulas:
            if (
                getattr(getattr(atom, "functor", None), "name", None)
                != self._type_predicate
                or not all(isinstance(a, Symbol) for a in atom.args)
            ):
                continue
            v, k = atom.args
            lookups = [
                f for f in formulas
                if f is not atom and k in f._symbols and v not in f._symbols
            ]
            if not lookups or any(
                set(extract_logic_free_variables(f)) != {k} for f in lookups
            ):
                continue
            body = lookups[0] if len(lookups) == 1 else Conjunction(
                tuple(lookups)
            )
            rows = self._closed_world_rows(body, (k,))
            if not rows or any(row[0] not in self._class_names for row in rows):
                continue
            rules = [
                Implication(
                    rule.consequent,
                    Conjunction(tuple(
                        Symbol(self._class_names[iri])(v) if f is atom
                        else ReplaceExpressionWalker(
                            {k: Constant[str](iri)}
                        ).walk(f)
                        for f in formulas
                    )),
                )
                for iri, in sorted(rows)
            ]
            if len(rules) == 1:
                return rules[0], []
            head = rule.consequent
            if not all(isinstance(a, Symbol) for a in head.args):
                continue
            union = Symbol.fresh()
            return Implication(head, union(*head.args)), [
                Implication(union(*head.args), r.antecedent) for r in rules
            ]
        return rule, []

    @property
    def current_program(self) -> List[fe.Expression]:
        """
        Returns the list of Front End Expressions that have
        currently been declared in the program, or through
        the program's constraints

        Returns
        -------
        List[fe.Expression]
            see description

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> P = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("b",), ("c",)], name="P"
        ... )
        >>> Q = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("d",), ("c",)], name="Q"
        ... )
        >>> with nl.scope as e:
        ...     e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        ...     cp = nl.current_program
        >>> cp
        [
            Z(x, ( PROB(x) )) ← ( P(x) ) ∧ ( Q(x) )
        ]
        """
        cp = []
        for constraint in self.program_ir.constraints().formulas:
            cp.append(self.frontend_translator.walk(constraint))
        cp += super().current_program
        return cp

    def _execute_query(
        self,
        head: typing.Union[fe.Symbol, Tuple[fe.Expression, ...]],
        predicate: fe.Expression,
        show_rewritten: bool = False,
        dry_run: bool = False,
        show_ra=False,
    ) -> Tuple[AbstractSet, Optional[ir.Symbol]]:
        """
        [Internal usage - documentation for developers]

        Performs an inferential query: will return as first output
        an AbstractSet with as many elements as solutions
        of the predicate query. AbstractSet's columns correspond to the
        expressions in the head.
        Typically, probabilities are abstracted and processed similar
        to symbols, though of different nature (see examples)
        If head expressions are arguments of a functor, the latter will
        be returned as the second output, defaulted as None

        Parameters
        ----------
        head : typing.Union[
            fe.Symbol,
            Tuple[fe.Expression, ...],
        ]
            see description
        predicate : Expression
            see description
        show_rewritten : bool
            If True, print the Datalog program after magic-sets rewriting.
        dry_run : bool
            If True, print the rewritten program and return immediately
            without running the chase.
        show_ra : bool or ShowRAOptions
            If True, print the named relational algebra expression
            of each rule instead of evaluating it. A
            :class:`~neurolang.frontend.show_ra.ShowRAOptions` selects
            how the plans are displayed.

        Returns
        -------
        Tuple[AbstractSet, Optional[ir.Symbol]]
            see description

        Examples
        --------
        Note: example ran with pandas backend
        >>> nl = NeurolangPDL()
        >>> P = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("b",), ("c",)], name="P"
        ... )
        >>> Q = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("d",), ("c",)], name="Q"
        ... )
        >>> with nl.scope as e:
        ...     e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        ...     s1 = nl._execute_query(tuple(), e.Z[e.x, e.p])
        ...     s2 = nl._execute_query((e.x, ), e.Z[e.x, e.p])
        ...     s3 = nl._execute_query(e.Z[e.x, e.p], e.Z[e.x, e.p])
        >>> s1
        (
            C{
                Empty DataFrame
                Columns: []
                Index: [0, 1]
                : typing.AbstractSet
            },
            None
        )
        >>> s2
        (
            C{
                    x
                0   a
                1   c
                : typing.AbstractSet
            },
            None
        )
        >>> s3
        (
            C{
                    x   p
                0   a   0.111111
                1   c   0.111111
                : typing.AbstractSet
            },
            S{Z: Unknown}
        )
        """
        if not self._bridging:
            bridges = self._type_bridges()
            if bridges:
                self._bridging = True
                try:
                    with self.scope:
                        self.program_ir.walk(Union(bridges))
                        return self._execute_query(
                            head, predicate, show_rewritten=show_rewritten,
                            dry_run=dry_run, show_ra=show_ra,
                        )
                finally:
                    self._bridging = False
        if _has_probability(head):
            return self._execute_probabilistic_head_query(
                head, predicate, show_rewritten=show_rewritten,
                dry_run=dry_run, show_ra=show_ra,
            )
        query_pred_symb = predicate.expression.functor
        query_pred_args = predicate.expression.args
        if is_probabilistic_predicate_symbol(query_pred_symb, self.program_ir):
            raise self._probabilistic_query_error(predicate.expression)

        # The probabilistic query path (below) assumes the predicate functor
        # maps to a rule entry (Union.formulas) in the program_ir symbol
        # table.  This holds when querying an IDB predicate defined by a
        # rule, but fails for:
        #   1. EDB predicates — symbol entry is a Constant (no .formulas).
        #   2. Conjunction queries (p(x), q(y)) — functor is `and_`, which
        #      is not in the symbol table at all.
        #   3. Heads computing a term, such as count(s): the path below
        #      only restricts the predicate's relation to its variables.
        # Conjunction queries, and EDB predicates when the program has an
        # ontology's constraints, are queried through a fresh rule defining
        # them, which takes the probabilistic path. The other cases fall
        # through to the base QueryBuilderDatalog implementation, which
        # creates a fresh query rule and only runs the deterministic chase.
        # A predicate the ontology constrains, as ``headOf`` by
        # ``Dean(x) -> exists y. headOf(x, y)``, is queried as an EDB one:
        # its constraints are not rules to solve, they rewrite the rule.
        symbol_entry = self.program_ir.symbol_table.get(query_pred_symb, None)
        if any(
            isinstance(rule, RightImplication)
            for rule in iter_implications(symbol_entry)
        ):
            symbol_entry = None
        if _solved_as_query_rule(head, symbol_entry):
            is_edb = symbol_entry is None or isinstance(
                symbol_entry, ir.Constant
            )
            if is_edb and (
                "__constraints__" in self.symbol_table or (
                    symbol_entry is None and
                    self._depends_on_probabilistic_queries(predicate)
                )
            ):
                # The deterministic chase of the base implementation
                # neither solves the probabilistic queries a conjunction
                # may depend on nor applies the ontology's constraints.
                if _computes_terms(head):
                    return self._execute_query_computing_terms(
                        head, predicate, show_rewritten, dry_run, show_ra
                    )
                return self._execute_query_through_rule(
                    head, predicate, show_rewritten, dry_run, show_ra
                )
            if not is_edb and self._depends_on_probabilistic_queries(
                predicate
            ):
                return self._execute_query_computing_terms(
                    head, predicate, show_rewritten, dry_run, show_ra
                )
            return super()._execute_query(
                head, predicate, show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )

        with self.scope:
            query, query_pred_args = self._query_rule(
                symbol_entry, predicate.expression
            )
            return self._solve_query_rule(
                head, query, query_pred_args, show_rewritten, dry_run, show_ra
            )

    def _query_rule(self, symbol_entry, query_atom):
        """
        Return the rule to solve for ``query_atom`` and its head's arguments.

        When the arguments of ``query_atom`` are variables and constants,
        not all distinct variables, the rule is a fresh one,
        ``q(v) :- query_atom`` with ``v`` the variables of ``query_atom``,
        which are returned, added to the program: magic sets then restrict
        the solved rules to the constants and repeated variables of
        ``query_atom``. Otherwise the rule is the first of
        ``symbol_entry``, the rules of the query predicate, and the
        arguments are those of ``query_atom``.
        """
        args = query_atom.args
        variables = tuple(
            dict.fromkeys(arg for arg in args if isinstance(arg, ir.Symbol))
        )
        if len(variables) == len(args) or not all(
            isinstance(arg, (ir.Symbol, ir.Constant)) for arg in args
        ):
            return symbol_entry.formulas[0], args
        goal = ir.Symbol.fresh()
        self.program_ir.walk(Implication(goal(*variables), query_atom))
        return self.program_ir.symbol_table[goal].formulas[0], variables

    def _solve_query_rule(
        self, head, query, query_pred_args, show_rewritten, dry_run, show_ra
    ):
        """
        Solve the rule ``query`` and restrict its solution to ``head``.

        ``query_pred_args`` are the arguments of the query predicate, the
        consequent of ``query``; the other parameters and the result are
        those of :meth:`_execute_query`.
        """
        options = show_ra_options(show_ra)
        if options is not None:
            return self._show_ra_query(
                query, options, show_rewritten, dry_run,
                (self._query_head_symbols(head)[0], query_pred_args),
            )

        solved = self._solve_rewritten(query, show_rewritten, dry_run)
        if solved is None:
            return ir.Constant(WrappedRelationalAlgebraFrozenSet()), None
        solution, query_pred_symb = solved

        head_symbols, functor_orig = self._query_head_symbols(head)
        solution = self._restrict_to_query_solution(
            head_symbols, query_pred_symb, query_pred_args, solution
        )
        if functor_orig is None:
            solution = solution.value
        return solution, functor_orig

    def _solve_rewritten(self, query, show_rewritten, dry_run):
        """
        Solve ``query``, rewritten with magic sets if it can be.

        The rewritten program is printed, within the scope holding the
        magic rules it needs, when ``show_rewritten`` or ``dry_run`` is
        set. When the query cannot be rewritten, ``query`` itself is
        printed and solved. Return the solution and the symbol of the
        solved query predicate, or None for a dry run.
        """
        try:
            magic_query, magic_rules = self._magic_rewrite_query(query)
            with self.scope:
                self.program_ir.walk(magic_rules)
                if self._handle_rewritten_output(
                    magic_query, show_rewritten, dry_run
                ):
                    return None
                solution = self._solve(magic_query)
                return solution, magic_query.consequent.functor
        except (InvalidMagicSetError, UnsupportedProgramError, SymbolNotFoundError):
            if self._handle_rewritten_output(
                query, show_rewritten, dry_run
            ):
                return None
            return self._solve(query), query.consequent.functor

    def _execute_probabilistic_head_query(self, head, predicate, **options):
        """
        Answer a query whose head has a probability, as ``ans(x, PROB(x))``.

        Only a rule computes a probability: the query becomes the rule
        ``q(x, PROB(x)) :- predicate`` of a fresh ``q``, which is queried
        in a scope closed on return, and whose columns are named after
        the query's head.
        """
        if isinstance(predicate, Condition):
            # a conditional body reaches the query as a Condition of
            # front-end expressions; '//' builds it as a rule body
            predicate = predicate.conditioned // predicate.conditioning
        variables = tuple(self.new_symbol() for _ in head)
        with self.scope:
            rule_symbol = self.new_symbol()
            self._declare_implication(rule_symbol(*head), predicate)
            solution, _ = self._execute_query(
                variables, rule_symbol(*variables), **options
            )
        if isinstance(solution, ir.Constant):
            # nothing was solved: the rules or their plans were printed
            return solution, None
        names = self._parameter_names(h.expression for h in head)
        return solution.rename_columns(
            {v.expression.name: name for v, name in zip(variables, names)}
        ), None

    def _solve_queries(self, queries, show_rewritten=False, dry_run=False):
        """
        Solve the *queries* together, each rewritten with magic sets for
        its own bindings: the program of all of them is stratified and
        solved once, so that what they share is computed once.

        Each query is the rule ``fresh(head) :- body`` of a fresh
        predicate, as in :meth:`_execute_query_computing_terms`, so that
        heads computing terms and probabilities are solved as any other.
        See :meth:`QueryBuilderDatalog._solve_queries`.
        """
        with self.scope:
            prepared = []
            for head, body in queries:
                if is_probabilistic_predicate_symbol(
                    getattr(body.expression, "functor", None),
                    self.program_ir
                ):
                    raise self._probabilistic_query_error(body.expression)
                fresh = self.new_symbol()
                self._declare_query(fresh(*head), body)
                variables = tuple(self.new_symbol() for _ in head)
                rule = self.program_ir.symbol_table[
                    fresh.expression
                ].formulas[0]
                prepared.append((head, variables, rule))
            rewritten = [
                self._rewrite_or_keep(rule) for _, _, rule in prepared
            ]
            with self.scope:
                self.program_ir.walk(
                    self._union_of_rules(rules for _, rules in rewritten)
                )
                goals = [goal for goal, _ in rewritten]
                if self._handle_rewritten_output(
                    goals, show_rewritten, dry_run
                ):
                    return None
                solution = self._solve(goals)
            return [
                self._joint_answer(head, variables, rule, goal, solution)
                for (head, variables, rule), (goal, _)
                in zip(prepared, rewritten)
            ]

    def _rewrite_or_keep(self, query):
        """
        Return ``query`` rewritten with magic sets, and the magic rules it
        needs, or ``query`` itself, without rules, if it cannot be.
        """
        try:
            return self._magic_rewrite_query(query)
        except (
            InvalidMagicSetError, UnsupportedProgramError, SymbolNotFoundError
        ):
            return query, Union(())

    def _joint_answer(self, head, variables, rule, goal, solution):
        """
        Return the answer of the query *head*, whose rule is *rule*,
        rewritten as *goal*, from the *solution* of the queries.
        """
        symbols = tuple(v.expression for v in variables)
        answer = self._restrict_to_query_solution(
            symbols, goal.consequent.functor, symbols, solution
        )
        names = self._parameter_names(h.expression for h in head)
        answer = answer.value.rename_columns(
            dict(zip(answer.value.columns, names))
        )
        if len(head) == 0:
            return len(answer) > 0
        return answer

    def _depends_on_probabilistic_queries(self, predicate):
        """Whether the rules ``predicate`` depends on have probabilistic
        queries, which the deterministic chase does not solve."""
        with self.scope:
            try:
                idbs = stratify_program(
                    self._declare_fresh_rule(predicate), self.program_ir
                )
            except SymbolNotFoundError:
                # Left to the base implementation, which reports it.
                return False
        return bool(idbs.get("probabilistic", Union(tuple())).formulas)

    def _probabilistic_query_error(self, atom):
        """
        Return the error of a query on the probabilistic predicate of
        ``atom``, naming the probabilistic predicates the query uses
        outside PROB[...] and writing, from the query's atoms, the
        PROB[...] queries to use instead.

        ``atom`` is the queried atom, or that of the fresh rule the query
        is solved through, whose body has the atoms of the query.
        """
        body = atom
        entry = self.program_ir.symbol_table.get(atom.functor, None)
        if atom.functor.is_fresh and hasattr(entry, "formulas"):
            body = entry.formulas[0].antecedent
        atoms = [
            a for a in extract_logic_atoms(body)
            if is_probabilistic_predicate_symbol(a.functor, self.program_ir)
        ] or [atom]
        names = list(dict.fromkeys(a.functor.name for a in atoms))
        one = len(names) == 1
        queries = list(filter(None, (_prob_query((a,)) for a in atoms)))
        message = (
            "Queries on probabilistic predicates are not supported: "
            f"{', '.join(names)} {'is' if one else 'are'} probabilistic. "
        )
        if queries:
            message += (
                f"To list {'its' if one else 'their'} tuples with their "
                f"probabilities, write {' or '.join(queries)}; to"
            )
        else:
            message += "To"
        message += (
            f" use {'it' if one else 'them'} in a condition, put the whole "
            "condition inside PROB[...]"
        )
        condition = _prob_query(
            body.formulas if isinstance(body, Conjunction) else ()
        )
        if condition:
            message += f", as in {condition}"
        return UnsupportedQueryError(message + ".")

    def _execute_query_through_rule(
        self, head, predicate, show_rewritten, dry_run, show_ra
    ):
        """
        Run the query of ``predicate`` through a fresh rule defining it.

        The rule ``fresh(x, ...) :- predicate``, over the free variables
        of ``predicate`` in ``head``, lives in a scope closed on return;
        querying it rewrites the program with the ontology's constraints,
        whose existentials only rewrite atoms of variables not in its head.
        """
        with self.scope:
            rule = self._declare_fresh_rule(predicate, head)
            return self._execute_query(
                head, fe.Expression(self, rule.consequent),
                show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )

    def _execute_query_computing_terms(
        self, head, predicate, show_rewritten, dry_run, show_ra
    ):
        """
        Run a query whose head computes terms, such as the aggregate
        ``max(p)``, over probabilistic queries: the deterministic chase of
        the base implementation does not solve them, the rule
        ``fresh(head) :- predicate`` is solved as any other.
        """
        with self.scope:
            fresh = self.new_symbol()
            self._declare_query(fresh(*head), predicate)
            variables = tuple(self.new_symbol() for _ in head)
            solution, _ = self._execute_query(
                variables, fresh(*variables),
                show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )
        if isinstance(solution, ir.Constant):
            # Nothing was run (dry_run, show_ra).
            return solution, None
        names = self._parameter_names(h.expression for h in head)
        return solution.rename_columns(
            dict(zip(solution.columns, names))
        ), None

    def _declare_fresh_rule(self, predicate, head=None):
        """Declare, in the current scope, the rule ``fresh(x, ...) :-
        predicate`` over the free variables of ``predicate``, those in the
        query's ``head`` if given, and return it as stored in the program."""
        variables = extract_logic_free_variables(predicate.expression)
        if head is not None:
            variables &= set().union(*(
                h._symbols for h in self._query_head_symbols(head)[0]
            ))
        variables = tuple(sorted(variables, key=lambda symbol: symbol.name))
        fresh = ir.Symbol.fresh()
        self.program_ir.walk(Union((
            Implication(fresh(*variables), predicate.expression),
        )))
        return self.program_ir.symbol_table[fresh].formulas[0]

    @staticmethod
    def _query_head_symbols(head):
        """
        Return the symbols of a query's head and its functor, if any.

        ``head`` is a tuple of expressions, or a predicate such as
        ``r(x, y)``, whose functor is returned (None for a tuple).
        """
        if not isinstance(head, tuple):
            # assumes head is a predicate e.g. r(x, y)
            return tuple(head.expression.args), head.expression.functor
        return tuple(t.expression for t in head), None

    def _magic_rewrite_query(self, query):
        """
        Rewrite ``query`` with magic sets.

        The rewriting is done in a scope that is closed on return, so the
        program is left unchanged. Return the rewritten query and the
        magic rules it needs.
        """
        try:
            return self._magic_rewrite_query_seeded(query, True)
        except (
            NewRecursionInMagicSetsRewriteError, NoConstantPredicateFoundError
        ):
            # not seeding only binds less: it would bind nothing either,
            # and it would not undo a recursion of the rewrite's own
            raise
        except (
            InvalidMagicSetError, UnsupportedProgramError, SymbolNotFoundError
        ):
            return self._magic_rewrite_query_seeded(query, False)

    def _magic_rewrite_query_seeded(self, query, seed_from_edb):
        """
        Rewrite ``query`` with magic sets, the literals of the extensional
        database with a constant binding variables if ``seed_from_edb``
        (see :class:`~neurolang.datalog.magic_sets.LeftToRightSIPS`).
        """
        with self.scope:
            goal, magic_rules = magic_rewrite(
                query.consequent, self.program_ir, seed_from_edb
            )
            self.program_ir.walk(magic_rules)
            magic_query = self.program_ir.symbol_table[goal].formulas[0]
            return probabilistic_postprocess_magic_rules(
                self.program_ir, magic_query, magic_rules
            )

    def _show_ra_query(
        self, query, options, show_rewritten, dry_run, query_atom
    ):
        """
        Print the RA plans of ``query`` instead of solving it.

        As in :meth:`_execute_query`, the plans are those of the query
        rewritten with magic sets, or of ``query`` itself when the rewritten
        query cannot be solved, followed by the restriction of the query
        predicate's relation to the query, whose head symbols and query
        predicate arguments are ``query_atom``. Errors raised while
        building the plans of ``query`` reach the caller. ``options`` is
        the :class:`~neurolang.frontend.show_ra.ShowRAOptions` of the
        display. An empty set is returned.
        """
        stratum_solver = ShowRAStratumSolver(
            self, options, generated_query_symbol(query)
        )
        try:
            magic_query, magic_rules = self._magic_rewrite_query(query)
            query_pred_symb = stratum_solver.query_predicate(
                query, magic_query
            )
            with self.scope:
                self.program_ir.walk(magic_rules)
                self._handle_rewritten_output(
                    magic_query, show_rewritten, dry_run, show_ra=True
                )
                solution = self._solve(
                    magic_query, stratum_solver=stratum_solver
                )
        except (InvalidMagicSetError, UnsupportedProgramError, SymbolNotFoundError):
            self._handle_rewritten_output(
                query, show_rewritten, dry_run, show_ra=True
            )
            solution = self._solve(query, stratum_solver=stratum_solver)
            query_pred_symb = query.consequent.functor
        head_symbols, query_pred_args = query_atom
        stratum_solver.print_query(
            head_symbols, query_pred_symb(*query_pred_args), solution
        )
        return ir.Constant(WrappedRelationalAlgebraFrozenSet()), None

    def solve_all(self, show_ra=False) -> Dict[str, NamedRelationalAlgebraFrozenSet]:
        """
        Returns a dictionary of "predicate_name": "Content"
        for all elements in the solution of the Datalog program.
        Typically, probabilities are abstracted and processed similar
        to symbols, though of different nature (see examples)

        Parameters
        ----------
        show_ra : bool or ShowRAOptions, optional
            If True, print the named RA expression of each rule instead
            of evaluating it, and return an empty solution. A
            :class:`~neurolang.frontend.show_ra.ShowRAOptions` selects how
            the plans are displayed.

        Returns
        -------
        Dict[str, NamedRelationalAlgebraFrozenSet]
            extensional and intentional facts that have been derived
            through the current program, optionally with probabilities

        Examples
        --------
        Note: example ran with pandas backend
        >>> nl = NeurolangPDL()
        >>> P = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("b",), ("c",)], name="P"
        ... )
        >>> Q = nl.add_uniform_probabilistic_choice_over_set(
        ...     [("a",), ("d",), ("c",)], name="Q"
        ... )
        >>> with nl.scope as e:
        ...     e.Z[e.PROB[e.x], e.x] = P[e.x] & Q[e.x]
        ...     solution = nl.solve_all()
        """
        solution_ir = self._solve(show_ra=show_ra)
        return self._solution_to_named_sets(solution_ir)

    def _solve(self, query=None, show_ra=False, stratum_solver=None):
        """
        Solve the program, restricted to ``query`` when given.

        The strata are solved by ``stratum_solver``; when it is None, a
        :class:`StratumSolver` or, if ``show_ra`` requests it, a
        :class:`~neurolang.frontend.show_ra.ShowRAStratumSolver`.
        """
        idbs = stratify_program(query, self.program_ir)
        det_idb = idbs.get("deterministic", Union(tuple()))
        prob_idb = idbs.get("probabilistic", Union(tuple()))
        postprob_idb = idbs.get("post_probabilistic", Union(tuple()))
        if stratum_solver is None:
            stratum_solver = self._stratum_solver(show_ra_options(show_ra))
        solution = stratum_solver.deterministic(det_idb)
        if prob_idb.formulas:
            solution = stratum_solver.probabilistic(solution, prob_idb)
        if postprob_idb.formulas:
            solution = stratum_solver.postprobabilistic(
                solution, postprob_idb
            )
        return solution

    def _stratum_solver(self, show_ra_opts):
        """Return the stratum solver for the ShowRAOptions (or None)."""
        if show_ra_opts is None:
            return StratumSolver(self)
        return ShowRAStratumSolver(self, show_ra_opts)

    def _solve_deterministic_stratum(self, det_idb):
        '''Resolution of the deterministic stratum. In case there
        are entries in the symbol table under the key __constraints__,
        a rewrite is performed and the resulting program is assigned
        to the variable `current_program_rewritten` to provide a way
        to access this information.

        In case connector_symbol has been defined, all rules derived
        from the ontology parsing are included in the program

        Parameters
        ----------
        det_idb : typing.Union
            union of rules composing the deterministic stratum.

        Returns
        -------
        Result obtained after resolution of the deterministic stratum
        '''
        added = set()
        if "__constraints__" in self.symbol_table:
            symbol_table = self.program_ir.symbol_table
            holds = frozenset(
                symbol for symbol in self.program_ir.constraints()._symbols
                if not _only_constraints(symbol_table.get(symbol, Union(())))
            )
            heads = {rule.consequent.functor for rule in det_idb.formulas}
            det_idb = self._rewrite_det_idb(det_idb, holds)
            # The predicates of the chased constraints and their copies.
            added = {
                rule.consequent.functor for rule in det_idb.formulas
            } - heads
            if self.connector_symbol is not None:
                connector_rules = tuple(
                    [q
                    for q in get_list_of_intensional_rules(self.program_ir)
                    if self.connector_symbol.expression in q.antecedent._symbols
                ])
                det_idb = Union(det_idb.formulas + connector_rules)
            self.current_program_rewritten = det_idb
        chase = self.chase_class(self.program_ir, rules=det_idb)
        solution = chase.build_chase_solution()
        for symbol in added & solution.keys():
            del solution[symbol]
        return solution

    def _solve_probabilistic_stratum(self, solution, prob_idb):
        CheckPredicateArities(self.program_ir).walk(prob_idb)
        pfact_edb = self.program_ir.probabilistic_facts()
        pchoice_edb = self.program_ir.probabilistic_choices()
        det_solution = with_empty_relations(
            solution, prob_idb, self.program_ir
        )
        for i, (succ_solver, marg_solver) in enumerate(
            zip(self.probabilistic_solvers, self.probabilistic_marg_solvers)
        ):
            try:
                prob_solution = compute_probabilistic_solution(
                    det_solution,
                    pfact_edb,
                    pchoice_edb,
                    prob_idb,
                    succ_solver,
                    marg_solver,
                    self.check_qbased_pfact_tuple_unicity,
                    self.program_ir.builtins(),
                )
            except UnsupportedSolverError:
                if i == len(self.probabilistic_solvers) - 1:
                    raise
            else:
                break

        wlq_symbs = set(
            rule.consequent.functor
            for rule in prob_idb.formulas
            if is_within_language_prob_query(rule)
        )
        solution.update(
            {
                pred_symb: relation
                for pred_symb, relation in prob_solution.items()
                if pred_symb in wlq_symbs
            }
        )
        return solution

    def _build_postprobabilistic_solver(self, solution, postprob_idb):
        """
        Build the solver of the post-probabilistic deterministic stratum.

        The relations of ``solution`` become extensional predicates, the
        builtins of the program are copied, and the rules of
        ``postprob_idb`` are added.
        """
        solver = RegionFrontendCPLogicSolver()
        for psymb, relation in solution.items():
            solver.add_extensional_predicate_from_tuples(
                psymb,
                relation.value,
            )
        for builtin_symb in self.program_ir.builtins():
            solver.symbol_table[builtin_symb] = self.program_ir.symbol_table[
                builtin_symb
            ]
        solver.walk(postprob_idb)
        return solver

    def _solve_postprobabilistic_deterministic_stratum(
        self, solution, postprob_idb, make_chase=None
    ):
        """
        Solve the post-probabilistic deterministic stratum.

        The rules of ``postprob_idb`` are run over ``solution`` by the
        chases ``make_chase(program, rules)`` returns, by default chases
        of the engine's chase class.
        """
        if make_chase is None:
            def make_chase(program, rules):
                return self.chase_class(program, rules=rules)
        solver = self._build_postprobabilistic_solver(solution, postprob_idb)
        chase = make_chase(solver, postprob_idb)
        solution = chase.build_chase_solution()
        return solution

    @staticmethod
    def _restrict_to_query_solution(
        head_symbols, pred_symb, pred_args, solution
    ):
        """
        Based on a solution instance and a query predicate symbol and args,
        retrieve the relation whose columns correspond to symbols in the head
        of the query.
        """
        # return an empty set when empty solution (reported in GH481)
        if pred_symb not in solution:
            return ir.Constant[AbstractSet](
                NamedRelationalAlgebraFrozenSet(
                    _symbol_names(head_symbols), []
                )
            )
        query_solution = solution[pred_symb].value.unwrap()
        query_row_type = solution[pred_symb].value.row_type
        constant_selection = {
            i: c.value for i, c in enumerate(pred_args)
            if isinstance(c, ir.Constant)
        }
        if constant_selection:
            query_solution = query_solution.selection(constant_selection)
        if len(pred_args) == len(head_symbols):
            cols = list(
                arg.name if isinstance(arg, ir.Symbol) else symb.name
                for arg, symb in zip(pred_args, head_symbols)
            )
        else:
            cols = list(
                arg.name if isinstance(arg, ir.Symbol)
                else ir.Symbol.fresh().name
                for arg in pred_args
            )
        query_solution = NamedRelationalAlgebraFrozenSet(cols, query_solution)
        query_solution = query_solution.projection(
            *(symb.name for symb in head_symbols)
        )
        type_args = get_args(query_row_type)
        proj_row_type = tuple(
            type_args[cols.index(symb.name)] for symb in head_symbols
        )
        query_solution.row_type = replace_type_variable_fix_python36_37(
            query_row_type, proj_row_type
        )
        return ir.Constant[AbstractSet](query_solution)

    @instance_lru_cache(
        key_fn=lambda det_idb, holds: (
            frozenset(str(r) for r in det_idb.formulas), holds
        ),
        maxsize=32, valid_while=_constraints_of,
    )
    def _rewrite_det_idb(self, det_idb, holds):
        """Xrewrite det_idb against ontology constraints, LRU-cached per instance.

        Only the constraints whose antecedent can hold, from their
        predicates ``holds`` with facts or rules, are used.

        Only the Xrewrite step is cached. The connector_rules scan that follows
        in _solve_deterministic_stratum accesses live program_ir and is NOT
        cached, so EDB added after load_ontology() remains visible.
        Cache is invalidated when the constraints change (an ontology is
        loaded, a scope is popped).
        """
        return self._rewrite_program_with_ontology(det_idb, holds)

    @instance_lru_cache(
        key_fn=lambda holds: holds, maxsize=4, valid_while=_constraints_of
    )
    def _constraints_that_can_hold(self, holds):
        return constraints_that_can_hold(
            self.program_ir.get_constraints(), holds
        )

    @instance_lru_cache(
        key_fn=lambda holds: holds, maxsize=4, valid_while=_constraints_of
    )
    def _projecting_rewritings(self, holds):
        """The rewritings of the projecting constraints that can hold, the
        same for every query, which can have values, and the predicates
        with values with them."""
        constraints = self._constraints_that_can_hold(holds)
        rules, valued, _ = valued_rules(constraints, holds, OntologyRewriter(
            Union(projecting_rules(constraints)), constraints, chased=True
        ).Xrewrite())
        return tuple(rules), frozenset(valued)

    def _rewrite_program_with_ontology(self, deterministic_program, holds):
        """Xrewrite with the constraints that can hold, those without
        existential variables only on values they introduce; the chase
        applies the latter, as rules, to the named values, with the
        rewritings of those that project a variable out."""
        constraints = self._constraints_that_can_hold(holds)
        rewrite = OntologyRewriter(
            Union(sum((
                _split_aggregation(rule, constraints)
                for rule in deterministic_program.formulas
            ), ())),
            constraints, chased=True, holds=holds,
        ).Xrewrite()
        projecting, valued = self._projecting_rewritings(holds)
        return Union(chase_rules(
            constraints, valued, tuple(rewrite) + projecting,
            self.program_ir.extensional_database(),
        ))

    def add_probabilistic_facts_from_tuples(
        self,
        iterable: Iterable[Tuple[Any, ...]],
        type_: Type = ir.Unknown,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Add probabilistic facts from tuples whose first element
        contains the probability label attached to that tuple.
        In the tuple (p, a, b, ...), p is the float probability
        of tuple (a, b, ...) to be True in any possible world.

        Note that each tuple from the iterable is independent
        from the others, meaning that multiple tuples can be True
        in the same possible world, contrary to a probabilistic choice.
        See example for details.

        Warnings
        --------
        Typing for the iterable is improper, true -but yet unsupported
        in Python typing- typing should be Iterable[Tuple[float, Any, ...]]
        See examples

        Parameters
        ----------
        iterable : Iterable[Tuple[Any, ...]]
            the first float number represents the probability
            of the tuple constituted of the remaining elements
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default ir.Unknown
        name : Optional[str], optional
            name for the resulting fe.Symbol, if None
            will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> p = [(0.8, 'a', 'b'), (0.7, 'b', 'c')]
        >>> nl.add_probabilistic_facts_from_tuples(p, name="P")
        P: typing.AbstractSet[typing.Tuple[float, str]] = \
            [(0.8, 'a', 'b'), (0.7, 'b', 'c')]

        Adds the probabilistic facts:
            P((a, b)) : 0.8  <-  T
            P((b, c)) : 0.7  <-  T
        """
        return self._add_probabilistic_tuples(
            iterable,
            type_,
            name,
            self.program_ir.add_probabilistic_facts_from_tuples,
        )

    def add_probabilistic_choice_from_tuples(
        self,
        iterable: Iterable[Tuple[Any, ...]],
        type_: Type = ir.Unknown,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Add probabilistic choice from tuples whose first element
        contains the probability label attached to that tuple.
        In the tuple (p, a, b, ...), p is the float probability
        of tuple (a, b, ...) to be True and all remaining tuples to
        be False in any possible world.

        Note that, contrary to a list of probabilistic facts, this represents
        a choice among possible values for the predicate, meaning that tuples
        in the set are mutually exclusive.
        See example for details.

        Warnings
        --------
        Typing for the iterable is improper, true -but yet unsupported-
        typing should be Iterable[Tuple[float, Any, ...]]
        See examples

        Parameters
        ----------
        iterable : Iterable[Tuple[Any, ...]]
            the first float number represents the probability
            of the tuple constituted of the remaining elements
            Note that the float probabilities must sum to 1.
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default ir.Unknown
        name : Optional[str], optional
            name for the resulting fe.Symbol, if None
            will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description

        Raises
        ------
        DistributionDoesNotSumToOneError
            if float probabilities do not sum to 1.

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> p = [(0.8, 'a', 'b'), (0.2, 'b', 'c')]
        >>> nl.add_probabilistic_choice_from_tuples(p, name="P")
        P: typing.AbstractSet[typing.Tuple[float, str, str]] = \
            [(0.8, 'a', 'b'), (0.2, 'b', 'c')]

        Adds the probabilistic choice:
            P((a, b)) : 0.8  v  P((b, c)) : 0.2  <-  T
        """
        return self._add_probabilistic_tuples(
            iterable,
            type_,
            name,
            self.program_ir.add_probabilistic_choice_from_tuples,
        )

    def _add_probabilistic_tuples(
        self, iterable, type_, name, solver_add_method
    ):
        if isinstance(iterable, pd.DataFrame):
            # a set, whose rows (not column names) give the type
            iterable = self.program_ir.new_set(iterable.rename(
                columns={n: i for i, n in enumerate(iterable.columns)}
            ))
        if name is None:
            name = str(uuid1())
        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = ir.Symbol[AbstractSet[type_]](name)
        solver_add_method(symbol, iterable)
        return fe.Symbol(self, name)

    def add_resampling(
        self,
        name: str,
        choice: str,
        n: int = 1000,
        scheme: str = "bayesian",
        seed: int = 0,
    ) -> fe.Symbol:
        """
        Add *n* replicates of the probabilistic choice *choice*, resampling
        its outcomes, for confidence intervals on the probabilities over it
        and on the measures derived from them.

        The resampling is the relation ``name(b, c..., w)``: in replicate
        ``b`` the outcome ``c`` has ``w`` times its probability in the
        choice.  ``RESAMPLE[name](pred(x)) = b`` in a rule body is then
        ``pred(x)`` computed on replicate ``b``; every query over the same
        resampling sees the same replicates.

        Parameters
        ----------
        name : str
            name of the resampling relation.
        choice : str
            name of a probabilistic choice of the program.
        n : int, optional
            number of replicates, at least 1, by default 1000.
        scheme : str, optional
            ``"bayesian"`` (Dirichlet weights, the Bayesian bootstrap),
            ``"multinomial"`` (the classical bootstrap) or ``"poisson"``,
            by default ``"bayesian"``.
        seed : int, optional
            seed of the random draws, by default 0.

        Returns
        -------
        fe.Symbol
            the resampling relation.

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> nl.add_tuple_set([('a', 'x'), ('b', 'x')], name="R")
        >>> nl.add_uniform_probabilistic_choice_over_set(
        ...     [('a',), ('b',)], name="C"
        ... )
        >>> nl.add_resampling("boot", "C", n=200)
        >>> nl.execute_datalog_program('''
        ... p(y, PROB) :- R(c, y) & C(c)
        ... p_rep(b, y, p) :- RESAMPLE[boot](p(y, p)) = b
        ... ans(y, percentile(p, 2.5), percentile(p, 97.5)) :- p_rep(b, y, p)
        ... ''')
        """
        choice_symbol = ir.Symbol(choice)
        choices = self.program_ir.probabilistic_choices()
        if choice_symbol not in choices:
            raise ValueError(f"{choice} is not a probabilistic choice")
        if n < 1:
            raise ValueError(f"A resampling needs replicates, not n={n}")
        rows = choices[choice_symbol].value.as_pandas_dataframe()

        def compute():
            return resampling_relation(rows, n, scheme, seed)

        # The replicates follow from the choice and the draws only: a
        # program may keep them (``derived_tables``, an engine's build), as
        # it keeps the pairs of a Euclidean bound.
        derived = getattr(self.program_ir, "derived_tables", None)
        relation = compute() if derived is None else derived(
            "resampling", [rows], (n, scheme, seed), compute
        )
        symbol = self.add_tuple_set(relation, name=name)
        self.resamplings[name] = choice_symbol, relation_sums(relation, n)
        return symbol

    def _expand_resamplings(self, rule):
        """*rule* with its ``RESAMPLE`` atoms rewritten over the replicates,
        adding the rewritten rules they depend on to the program."""
        programs = {}

        def resampled_predicate(name, functor):
            if name.name not in self.resamplings:
                raise ValueError(f"{name.name} is not a resampling")
            if name not in programs:
                choice, sums = self.resamplings[name.name]
                arity = len(
                    self.program_ir.probabilistic_choices()[choice]
                    .value.columns
                ) - 1
                programs[name] = ResampledProgram(
                    self.program_ir.intensional_database(),
                    name, choice, name, sums, arity,
                    self._resampled_versions.setdefault(name.name, {}),
                )
            return programs[name].predicate(functor)

        rule = rewrite_resample_markers(rule, resampled_predicate)
        for program in programs.values():
            self.program_ir.walk(Union(program.rules))
        return rule

    def add_uniform_probabilistic_choice_over_set(
        self,
        iterable: Iterable[Tuple[Any, ...]],
        type_: Type = ir.Unknown,
        name: Optional[str] = None,
    ) -> fe.Symbol:
        """
        Add uniform probabilistic choice over values in the iterable.

        Every tuple in the iterable will be assigned the same probability to
        be True, with all remaining tuples False, in any possible world.

        Note that, contrary to a list of probabilistic facts, this represents
        a choice among possible values for the predicate, meaning that tuples
        in the set are mutually exclusive.

        See example for details.

        Parameters
        ----------
        iterable : Iterable[Tuple[Any, ...]]
            typically List[Tuple[Any, ...]], other types of Iterable
            will be cast as lists
        type_ : Type, optional
            type for resulting AbstractSet if None will be inferred
            from the data, by default ir.Unknown
        name : Optional[str], optional
            name for the resulting fe.Symbol, if None
            will be fresh, by default None

        Returns
        -------
        fe.Symbol
            see description

        Examples
        --------
        >>> nl = NeurolangPDL()
        >>> p = [('a',), ('b',)]
        >>> nl.add_uniform_probabilistic_choice_over_set(p, name="P")
        P: typing.AbstractSet[typing.Tuple[float, str]] = \
            [(0.5, 'a'), (0.5, 'b')]

        Adds the probabilistic choice:
            P(a) : 0.5  v  P(b) : 0.5  <-  T
        """
        if name is None:
            name = str(uuid1())
        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        if isinstance(iterable, pd.DataFrame):
            iterable = iterable.rename(
                columns={n: i for i, n in enumerate(iterable.columns)}
            )
            arity = len(iterable.columns)
        else:
            arity = len(next(iter(iterable)))
        symbol = ir.Symbol[AbstractSet[type_]](name)
        columns = tuple(ir.Symbol.fresh().name for _ in range(arity))
        ra_set = NamedRelationalAlgebraFrozenSet(columns, iterable)
        prob_col = ir.Symbol.fresh().name
        probability = 1 / len(ra_set)
        projections = collections.OrderedDict()
        projections[prob_col] = probability
        for col in columns:
            projections[col] = RelationalAlgebraColumnStr(col)
        ra_set = ra_set.extended_projection(projections)
        self.program_ir.add_probabilistic_choice_from_tuples(
            symbol,
            ra_set.projection_to_unnamed(*projections.keys())
        )
        return fe.Symbol(self, name)


def _prob_query(atoms):
    """
    Return the query ``ans(x, ..., p) :- PROB[atoms] = p`` of the free
    variables ``x, ...`` of ``atoms`` in Datalog, or None if it cannot be
    written: no atoms, or atoms not of named predicates over variables and
    constants of the query.
    """
    printable = all(
        isinstance(a, ir.FunctionApplication)
        and isinstance(a.functor, ir.Symbol) and not a.functor.is_fresh
        and all(
            isinstance(arg, ir.Constant)
            or (isinstance(arg, ir.Symbol) and not arg.is_fresh)
            for arg in a.args
        )
        for a in atoms
    )
    if not atoms or not printable:
        return None
    printer = DatalogPrettyPrinter()
    variables = [
        v.name for v in extract_logic_free_variables(Conjunction(tuple(atoms)))
    ]
    p = "probability" if "p" in variables else "p"
    return (
        f"`ans({', '.join(variables + [p])}) :- "
        f"PROB[{', '.join(printer.walk(a) for a in atoms)}] = {p}`"
    )


def _has_probability(head) -> bool:
    """Return whether the query *head*, a tuple, has a ``PROB(…)``."""
    return isinstance(head, tuple) and any(
        getattr(getattr(h, "expression", None), "functor", None) == PROB
        for h in head
    )


def _split_aggregation(rule, constraints):
    """``H(count(x)) :- B`` as ``H(count(x)) :- b(x)`` and ``b(x) :- B``
    when ``B`` has a predicate the ``constraints`` derive: the aggregate is
    then over the union of the rewritings of ``B``."""
    if not is_aggregation_rule(rule) or not any(
        getattr(atom.functor, "name", None) in constraints
        for atom in extract_logic_atoms(rule.antecedent)
    ):
        return (rule,)
    body = ir.Symbol.fresh()(*sorted(
        extract_logic_free_variables(rule.antecedent)
        & rule.consequent._symbols, key=lambda symbol: symbol.name
    ))
    return (
        Implication(rule.consequent, body),
        Implication(body, rule.antecedent),
    )


def _only_constraints(entry):
    """Whether the symbol-table ``entry`` is a union of constraints only."""
    return isinstance(entry, Union) and all(
        isinstance(rule, RightImplication) for rule in entry.formulas
    )


def _solved_as_query_rule(head, symbol_entry) -> bool:
    """
    Return whether a query is solved as a rule of its own.

    It is when its predicate is not defined by rules (*symbol_entry* is
    not a union of rules) or when its *head* computes a term other than
    a probability, such as the aggregate ``count(s)``.
    """
    if not hasattr(symbol_entry, "formulas"):
        return True
    return _computes_terms(head)


def _computes_terms(head) -> bool:
    """Whether the query's *head* computes a term other than a probability,
    such as the aggregate ``count(s)``."""
    return isinstance(head, tuple) and any(
        isinstance(h.expression, ir.FunctionApplication)
        and h.expression.functor != PROB
        for h in head
    )
