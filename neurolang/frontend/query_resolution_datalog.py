r"""
Query Builder Datalog
=====================
Complements QueryBuilderBase with query capabilities,
as well as Region and Neurosynth capabilities
"""
from collections import defaultdict
from ..datalog.magic_sets import magic_rewrite
from ..exceptions import (
    ForbiddenDisjunctionError, NeuroLangException, UnsupportedProgramError,
)
from typing import (
    AbstractSet,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import uuid1

import pandas as pd

from .. import datalog
from .. import expressions as ir
from .. import logic
from ..datalog.chase import Chase
from ..datalog.constraints_representation import RightImplication
from ..datalog.exceptions import InvalidMagicSetError
from ..datalog.expression_processing import (
    remove_conjunction_duplicates,
    TranslateToDatalogSemantics,
    reachable_code,
)
from ..datalog.expressions import predicate_identity
from ..logic.transformations import RemoveTrivialOperations
from ..type_system import Unknown
from ..utils import NamedRelationalAlgebraFrozenSet, RelationalAlgebraFrozenSet
from .datalog.pretty_printer import DatalogPrettyPrinter
from .datalog.squall_syntax_lark import (
    parser as squall_parser,
    EquiprobableChoiceDef,
    SquallProgram,
    WeightedChoiceDef,
)
from .datalog.standard_syntax import parser as datalog_parser
from .query_resolution import NeuroSynthMixin, QueryBuilderBase, RegionMixin
from ..datalog import DatalogProgram
from ..datalog.wrapped_collections import WrappedRelationalAlgebraFrozenSet
from ..logic.horn_clauses import Fol2DatalogTranslationException
from . import query_resolution_expressions as fe

__all__ = ["QueryBuilderDatalog"]


def _wrap_fol_error_for_squall(exc: Fol2DatalogTranslationException) -> Exception:
    orig = exc.__cause__
    if orig and "Variables in head" in str(orig):
        new_msg = (
            f"SQUALL semantic error: Variable mismatch between rule definition and query.\n\n"
            f"The rule head uses a variable that doesn't appear in the body, "
            f"or the query uses variables that don't match the rule head.\n\n"
            f"Common fixes:\n"
            f"1. Add explicit labels in rule definitions:\n"
            f"   'define as X for every Noun ?x that...' (not just 'every Noun')\n"
            f"2. Ensure query variables match rule head variables:\n"
            f"   'obtain every X (?x; ?y)' must match the rule head\n"
            f"3. Check that rule body variables are used in the head\n\n"
            f"Original error: {orig}"
        )
        exc.args = (new_msg,)
    return exc


class QueryBuilderDatalog(RegionMixin, NeuroSynthMixin, QueryBuilderBase):
    """
    Complements QueryBuilderBase with query capabilities,
    as well as Region and Neurosynth capabilities
    """

    def __init__(
        self,
        program_ir: DatalogProgram,
        chase_class: Type[Chase] = Chase,
    ) -> "QueryBuilderDatalog":
        """
        Query builder with query, Region, Neurosynth capabilities

        Parameters
        ----------
        program_ir : DatalogProgram
            Datalog program's intermediate representation,
            usually blank
        chase_class : Type[Chase], optional
            used to compute deterministic solutions,
            by default Chase

        Returns
        -------
        QueryBuilderDatalog
            see description
        """
        super().__init__(program_ir, logic_programming=True)
        self.chase_class = chase_class
        self.frontend_translator = fe.TranslateExpressionToFrontEndExpression(
            self
        )
        self.translate_expression_to_datalog = TranslateToDatalogSemantics()
        self.datalog_parser = datalog_parser

    @staticmethod
    def _simplify_rewritten_rules(reachable_rules):
        """Simplify reachable rules using existing expression walkers.

        Applies RemoveTrivialOperations (unwraps single-element n-ary
        operators, removes double negation) and remove_conjunction_duplicates
        (deduplicates body predicates), then deduplicates rules.

        Note: TRUE removal from conjunctions is not covered by
        the current MRO walkers — the LogicSimplifier in the SQUALL
        parser handles that but is not in the general MRO.
        """
        if reachable_rules is None:
            return None

        trivial = RemoveTrivialOperations()

        def _simplify_single_rule(rule):
            try:
                rule = trivial.walk(rule)
            except Exception:
                pass
            if isinstance(rule.antecedent, logic.Conjunction):
                try:
                    deduped = remove_conjunction_duplicates(rule.antecedent)
                    if deduped is not rule.antecedent:
                        rule = logic.Implication(rule.consequent, deduped)
                except Exception:
                    pass
            return rule

        simplified = [_simplify_single_rule(rule) for rule in reachable_rules.formulas]

        seen = set()
        unique = []
        for rule in simplified:
            key = repr(rule)
            if key not in seen:
                seen.add(key)
                unique.append(rule)

        return logic.Union(tuple(unique))

    def _print_rewritten_program(
        self, query_expression, reachable_rules=None
    ):
        """Print the Datalog program after magic-sets rewriting."""
        reachable_rules = self._simplify_rewritten_rules(reachable_rules)

        print("── rewritten program ──")
        printer = DatalogPrettyPrinter()
        print(printer.walk(query_expression))
        if reachable_rules is not None:
            for rule in reachable_rules.formulas:
                print(printer.walk(rule))
        print()

    def _handle_rewritten_output(
        self, query_expression, show_rewritten: bool, dry_run: bool
    ) -> bool:
        """Print rewritten program if requested and return True if dry-running.

        Returns True when dry_run is active (caller should short-circuit),
        False when execution should continue.
        """
        if show_rewritten or dry_run:
            reachable_rules = reachable_code(
                query_expression, self.program_ir
            )
            self._print_rewritten_program(
                query_expression, reachable_rules
            )
        return dry_run

    @property
    def current_program(self) -> List[fe.Expression]:
        """
        Returns the list of expressions that have currently been
        declared in the program

        Returns
        -------
        List[fe.Expression]
            see description

        Examples
        --------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     e.l2[e.x] = e.l[e.x, e.y] & (e.x == e.y)
        ...     cp = nl.current_program
        >>> cp
        [
            l2(x) ← ( l(x, y) ) ∧ ( x eq y )
        ]
        """
        cp = []
        for rules in self.program_ir.intensional_database().values():
            for rule in rules.formulas:
                cp.append(self.frontend_translator.walk(rule))
        return cp

    def _declare_implication(
        self, consequent: fe.Expression, antecedent: fe.Expression
    ) -> fe.Expression:
        """
        Creates an implication of the consequent by the antecedent
        and adds the rule to the current program:
            consequent <- antecedent

        Parameters
        ----------
        consequent : fe.Expression
            see description, will be processed to a logic form before
            creating the implication rule
        antecedent : fe.Expression
            see description, will be processed to a logic form before
            creating the implication rule

        Returns
        -------
        fe.Expression
            see description

        Examples
        --------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     nl._declare_implication(e.l2[e.x], e.l2[e.x, e.y])
        ...     cp = nl.current_program
        >>> cp
        [
            l2(x) ← l(x, y)
        ]
        """
        consequent = self.translate_expression_to_datalog.walk(
            consequent.expression
        )
        antecedent = self.translate_expression_to_datalog.walk(
            antecedent.expression
        )
        rule = datalog.Implication(consequent, antecedent)
        self.program_ir.walk(rule)
        return rule

    def add_constraint(
        self, antecedent: fe.Expression, consequent: fe.Expression
    ) -> fe.Expression:
        """
        Creates an right implication of the consequent by the antecedent
        and adds the rule to the current program:

        .. code-block:: text

            antecedent -> consequent

        Parameters
        ----------
        antecedent : fe.Expression
            see description, will be processed to a logic form before
            creating the right implication rule
        consequent : fe.Expression
            see description, will be processed to a logic form before
            creating the right implication rule

        Returns
        -------
        fe.Expression
            see description

        Examples
        --------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     nl.add_constraint(e.l2[e.x, e.y], e.l2[e.x])
        """
        consequent = self.translate_expression_to_datalog.walk(
            consequent.expression
        )
        antecedent = self.translate_expression_to_datalog.walk(
            antecedent.expression
        )
        rule = RightImplication(antecedent, consequent)
        self.program_ir.walk(rule)
        return rule

    def execute_datalog_program(
        self, code: str, show_rewritten: bool = False, dry_run: bool = False,
        show_ra: bool = False,
    ) -> Union[None, bool, RelationalAlgebraFrozenSet]:
        """
        Execute a Datalog program in classical syntax.
        If the program contains a query, in the form `ans(x) :- R(x)`,
        then this query is executed against the program and the result is
        returned. Otherwise returns None.

        Parameters
        ----------
        code : string
            Datalog program.
        show_rewritten : bool
            If True, print the Datalog program after magic-sets rewriting.
        dry_run : bool
            If True, print the rewritten program and return None without
            running the chase.

        Examples
        --------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> prog = '''
        ...        l2(x, y) :- l(x, y), (x == y)
        ...        ans(x) :- l2(x, y)
        ...        '''
        >>> with nl.environment as e:
        ...     q = nl.execute_datalog_program(prog)
        >>> q
        ... q: typing.AbstractSet[typing.Tuple[int]] = [(2,)]
        """
        intermediate_representation = self.datalog_parser(code)
        queries = [
            rule
            for rule in intermediate_representation.formulas
            if isinstance(rule, ir.Query)
        ]
        if len(queries) == 0:
            self.program_ir.walk(intermediate_representation)
            return
        elif len(queries) == 1:
            query = self.frontend_translator.walk(queries[0])
            program = logic.Union(
                [
                    rule
                    for rule in intermediate_representation.formulas
                    if not isinstance(rule, ir.Query)
                ]
            )
            self.program_ir.walk(program)
            return self.query(
                query.head.arguments, query.body,
                show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )
        else:
            raise UnsupportedProgramError(
                "Only one query, in the form of ans(...) :- R(...) is "
                "supported. Datalog program has more than one query rule: "
                "{}".format(
                    "\n".join(
                        [
                            str(self.frontend_translator.walk(q))
                            for q in queries
                        ]
                    )
                )
            )

    @staticmethod
    def _process_squall_commands(parsed):
        """Process SQUALL directives (#set_backend, etc.)."""
        if not (isinstance(parsed, SquallProgram) and parsed.commands):
            return
        from neurolang.config import config as nl_config
        from neurolang.expressions import Constant

        _KNOWN_COMMANDS = {"set_backend"}
        for cmd in parsed.commands:
            name = cmd.functor.name if hasattr(cmd, 'functor') else None
            if name is None or name not in _KNOWN_COMMANDS:
                continue
            if name == "set_backend":
                if cmd.args and isinstance(cmd.args[0], Constant):
                    nl_config.set_query_backend(cmd.args[0].value)
                else:
                    raise NeuroLangException(
                        "#set_backend requires a string argument, "
                        "e.g. #set_backend('pandas')."
                    )

    def _walk_squall_rule(self, rule):
        """Walk a single SQUALL rule or choice definition into the engine."""
        if isinstance(rule, EquiprobableChoiceDef):
            self._handle_equiprobable_choice(rule)
        elif isinstance(rule, WeightedChoiceDef):
            self._handle_weighted_choice(rule)
        else:
            try:
                self.program_ir.walk(rule)
            except Fol2DatalogTranslationException as e:
                raise _wrap_fol_error_for_squall(e) from e

    def _walk_squall_rules(self, parsed):
        """Walk all rule definitions from a parsed SQUALL program into the engine."""
        if not isinstance(parsed, SquallProgram):
            # Legacy non-SquallProgram parse tree
            if isinstance(parsed, (EquiprobableChoiceDef, WeightedChoiceDef)):
                self._walk_squall_rule(parsed)
            elif isinstance(parsed, logic.Union):
                for r in parsed.formulas:
                    self._walk_squall_rule(r)
            else:
                self._walk_squall_rule(parsed)
            return
        for rule in parsed.rules_and_choice_defs:
            self._walk_squall_rule(rule)

    def _execute_single_squall_query(self, query, rules_and_choice_defs,
                                     show_rewritten=False, dry_run=False,
                                     show_ra=False):
        """Execute one obtain clause from a SQUALL program.

        Builds a fresh helper predicate h(head_vars) :- query.body and
        delegates to _execute_query, exactly as execute_datalog_program
        does for ans(...) :- R(...).
        """
        head = query.head
        if isinstance(head, ir.FunctionApplication):
            head_vars = tuple(head.args)
        elif isinstance(head, ir.Symbol):
            head_vars = (head,)
        else:
            head_vars = tuple(head)

        h = ir.Symbol.fresh()
        query_impl = datalog.Implication(h(*head_vars), query.body)

        self.program_ir.push_scope()
        try:
            for rule in rules_and_choice_defs:
                if isinstance(rule, (EquiprobableChoiceDef, WeightedChoiceDef)):
                    continue  # already registered in global scope
                try:
                    self.program_ir.walk(rule)
                except ForbiddenDisjunctionError:
                    pass
            self.program_ir.walk(query_impl)
            fe_pred = fe.Expression(self, h(*head_vars))
            fe_head = tuple(
                fe.Expression(self, ir.Symbol(s.name))
                for s in head_vars
            )
            ra, _ = self._execute_query(
                fe_head, fe_pred,
                show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )
        finally:
            self.program_ir.pop_scope()
        return ra

    def execute_squall_program(
        self, code: str, show_rewritten: bool = False, dry_run: bool = False,
        show_ra: bool = False,
    ) -> Union[None, NamedRelationalAlgebraFrozenSet, Dict[str, NamedRelationalAlgebraFrozenSet]]:
        """
        Execute a SQUALL (controlled English) program.

        Parses *code* as a SQUALL program, walks rule definitions into the
        engine, and executes ``obtain`` queries the same way
        `execute_datalog_program` handles ``ans(x) :- R(x)`` — by building a
        fresh helper implication and delegating to `self.query` (or
        `self._execute_query` for probabilistic engines).

        Parameters
        ----------
        code : str
            SQUALL program text.  May contain any mixture of
            ``define as …`` rule definitions and ``obtain …`` queries.
        show_rewritten : bool
            If True, print the Datalog program after magic-sets rewriting.
        dry_run : bool
            If True, print the rewritten program and return None without
            running the chase.

        Returns
        -------
        None
            When *code* contains only rule definitions (no ``obtain``).
        NamedRelationalAlgebraFrozenSet
            When *code* contains exactly one ``obtain`` clause.
        Dict[str, NamedRelationalAlgebraFrozenSet]
            When *code* contains two or more ``obtain`` clauses, keyed
            ``"obtain_0"``, ``"obtain_1"``, … in declaration order.

        Examples
        --------
        >>> from neurolang.frontend import NeurolangPDL
        >>> nl = NeurolangPDL()
        >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
        >>> _ = nl.add_tuple_set([("alice",)], name="plays")
        >>> nl.execute_squall_program(
        ...     "define as Active every person that plays."
        ... ) is None
        True
        >>> result = nl.execute_squall_program("obtain every Person that plays.")
        >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
        ['alice']
        """
        parsed = squall_parser(code)

        self._process_squall_commands(parsed)

        # Rules-only (no obtain) — backward-compat: returns Union/Implication
        if not isinstance(parsed, SquallProgram):
            self._walk_squall_rules(parsed)
            return None

        # Walk all rule definitions into the engine (global scope).
        self._walk_squall_rules(parsed)

        if not parsed.queries:
            return None

        # Execute each obtain query
        results = {}
        for i, q in enumerate(parsed.queries):
            key = parsed.query_names.get(i, f"obtain_{i}")
            ra = self._execute_single_squall_query(
                q, parsed.rules_and_choice_defs,
                show_rewritten=show_rewritten, dry_run=dry_run,
                show_ra=show_ra,
            )
            if not dry_run:
                results[key] = ra

        if dry_run:
            return None
        if len(results) == 1:
            return next(iter(results.values()))
        return results

    def compute_datalog_program_for_autocompletion(
            self, code: str, autocompletion_code
    ) -> Dict:
        """
        Computes a Datalog program in classical syntax.
        Returns the next accepted tokens of the program.

        Parameters
        ----------
        code : string
            Datalog program.
        autocompletion_code : str
            Datalog program for autocompletion.

        Examples
        --------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> prog_complete = '''
        ...        l2(x, y) :- l(x, y), (x == y)
        ...        ans(x) :- l2(x, y)
        ...        '''
        >>> prog = '''
        ...        l2(x, y) :- l(x, y), (x == y)
        ...        ans(x) :-
        ...        '''
        >>> with nl.environment as e:
        ...     q = nl.compute_datalog_program_for_autocompletion(prog_complete, prog)
        >>> q
        ... {
        ...     'Signs': {'@', '∃', '('}, 'Numbers': set(), 'Text': set(), 'Operators': {'¬', '~'},
        ...     'Cmd_identifier': set(), 'Functions': {'lambda'}, 'Identifier_regexp': set(),
        ...     'Reserved words': {'exists', 'EXISTS'}, 'Boleans': {'⊤', '⊥', 'False', 'True'},
        ...     'Expression symbols': set(), 'Python string': set(),
        ...     'Strings': {'<identifier regular expression>', '<command identifier>'}, 'commands': set(),
        ...     'functions': set(), 'base symbols': set(), 'query symbols': set()
        ... }
        """
        # All the following code before return is mandatory to retrieve
        # the query symbols without actually running the query
        intermediate_representation = self.datalog_parser(code)
        queries = [
            rule
            for rule in intermediate_representation.formulas
            if isinstance(rule, ir.Query)
        ]
        if len(queries) == 1:
            self.frontend_translator.walk(queries[0])
            program = logic.Union(
                [
                    rule
                    for rule in intermediate_representation.formulas
                    if not isinstance(rule, ir.Query)
                ]
            )
            self.program_ir.walk(program)
        else:
            self.program_ir.walk(intermediate_representation)
        res = self.datalog_parser(autocompletion_code, None, None, True)
        return res

    def _handle_equiprobable_choice(self, choice_def):
        """Register a SQUALL equiprobable choice in the program IR.

        Looks up the source EDB set in the symbol table, computes uniform
        probabilities (1 / N) for each tuple, and calls
        ``add_probabilistic_choice_from_tuples``.
        """
        body_formula = choice_def.body_formula

        # Extract the source predicate symbol from the body formula.
        if isinstance(body_formula, logic.Conjunction):
            # Filtered source — extract first predicate as the base source.
            source_pred_sym = body_formula.formulas[0].functor
        elif isinstance(body_formula, ir.FunctionApplication):
            source_pred_sym = body_formula.functor
        else:
            raise NeuroLangException(
                f"Cannot extract source predicate from body formula: "
                f"{body_formula}"
            )

        # Look up the source EDB set in the program symbol table.
        if source_pred_sym not in self.program_ir.symbol_table:
            raise NeuroLangException(
                f"Source predicate '{source_pred_sym.name}' not found in the "
                f"program. Make sure the data is loaded before defining the "
                f"probabilistic choice."
            )
        source_rel = self.program_ir.symbol_table[source_pred_sym]
        if not isinstance(source_rel, ir.Constant):
            raise NeuroLangException(
                f"Source predicate '{source_pred_sym.name}' is not a "
                f"concrete set. Equiprobable choices currently require an "
                f"extensional (ground) source predicate."
            )

        # Build choice tuples with uniform probability.
        # source_rel.value is a WrappedRelationalAlgebraSet — use itervalues
        # to get raw Python tuples for each row.
        source_rows = list(source_rel.value.itervalues())
        n = len(source_rows)
        if n == 0:
            prob = 1.0
        else:
            prob = 1.0 / n
        choice_tuples = [(prob,) + row for row in source_rows]

        self.program_ir.add_probabilistic_choice_from_tuples(
            choice_def.head_symbol, choice_tuples
        )

    def _handle_weighted_choice(self, choice_def):
        """Register a SQUALL weighted probabilistic choice in the program IR.

        Evaluates the probability expression per tuple from the source set,
        normalises to sum to 1, and calls ``add_probabilistic_choice_from_tuples``.
        """
        raise NeuroLangException(
            "Weighted probabilistic choices (with explicit probability "
            "expressions) are not yet implemented. "
            "Use 'define as X as an equiprobable choice over every Y' for now."
        )

    def query(
        self, *args, show_rewritten: bool = False, dry_run: bool = False,
        show_ra: bool = False,
    ) -> Union[bool, RelationalAlgebraFrozenSet, fe.Symbol]:
        """
        Performs an inferential query on the database.
        There are three modalities
        1. If there is only one argument, the query returns `True` or `False`
        depending on whether the query could be inferred.
        2. If there are two arguments and the first is a tuple of `fe.Symbol`,
        it returns the set of results meeting the query in the second argument.
        3. If the first argument is a predicate (e.g. `Q(x)`) it performs the
        query, adds it to the engine memory, and returns the
        corresponding symbol.
        See example for 3 modalities

        Parameters
        ----------
        show_rewritten : bool
            If True, print the Datalog program after magic-sets rewriting.
        dry_run : bool
            If True, print the rewritten program and return immediately
            without running the chase.

        Returns
        -------
        Union[bool, RelationalAlgebraFrozenSet, fe.Symbol]
            read the descrpition.

        Examples
        --------
        Note: example ran with pandas backend
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.environment as e:
        ...     e.l2[e.x, e.y] = e.l[e.x, e.y] & (e.x == e.y)
        ...     s1 = nl.query(e.l2[e.x, e.y])
        ...     s2 = nl.query((e.x,), e.l2[e.x, e.y])
        ...     s3 = nl.query(e.l3[e.x], e.l2[e.x, e.y])
        """
        if len(args) == 1:
            predicate = args[0]
            head = tuple()
        elif len(args) == 2:
            head, predicate = args
            if (
                isinstance(head, (fe.Symbol, fe.Expression)) and
                isinstance(head.expression, ir.Symbol)
            ):
                head = (head,)
        else:
            raise ValueError("query takes 1 or 2 arguments")

        solution_set, functor_orig = self._execute_query(
            head, predicate, show_rewritten=show_rewritten, dry_run=dry_run,
            show_ra=show_ra,
        )

        if not isinstance(head, tuple):
            out_symbol = ir.Symbol[solution_set.type](functor_orig.name)
            self.add_tuple_set(solution_set.value, name=functor_orig.name)
            return fe.Symbol(self, out_symbol.name)
        elif len(head) == 0:
            return len(solution_set) > 0
        else:
            return solution_set

    def _magic_sets_rewrite_query(
        self,
        head: Union[fe.Symbol, Tuple[fe.Expression, ...]],
        predicate: fe.Expression,
    ) -> Tuple[logic.Implication, logic.Union]:
        """Build query expression and run magic-sets rewriting.

        Returns the rewritten query expression and the reachable rules.
        If magic-sets rewriting is not applicable, returns the original
        query expression and reachable rules.

        NOTE: The caller must have already created a sub-scope on the
        symbol table via ``self.program_ir.symbol_table =
        self.symbol_table.create_scope()`` and must restore it after
        the chase or printing is complete.

        Parameters
        ----------
        head : Union[fe.Symbol, Tuple[fe.Expression, ...]]
            Query head.
        predicate : fe.Expression
            Query body.

        Returns
        -------
        Tuple[logic.Implication, logic.Union]
            Rewritten query expression and reachable rules.

        """
        if isinstance(head, fe.Operation):
            new_head = self.new_symbol()(*head.arguments)
        elif isinstance(head, tuple):
            new_head = self.new_symbol()(*head)
        else:
            raise ValueError("Wrong head syntax")
        query_expression = self._declare_implication(new_head, predicate)

        try:
            magic_query_expression = self.magic_sets_rewrite_program(
                query_expression)
            reachable_rules = reachable_code(
                magic_query_expression, self.program_ir)
            return magic_query_expression, reachable_rules
        except InvalidMagicSetError:
            reachable_rules = reachable_code(query_expression, self.program_ir)
            return query_expression, reachable_rules

    def _execute_query(
        self,
        head: Union[fe.Symbol, Tuple[fe.Expression, ...]],
        predicate: fe.Expression,
        show_rewritten: bool = False,
        dry_run: bool = False,
        show_ra: bool = False,
    ) -> Tuple[AbstractSet, Optional[ir.Symbol]]:
        """
        [Internal usage - documentation for developers]

        Performs an inferential query. Will return as first output
        an AbstractSet with as many elements as solutions of the
        predicate query. The AbstractSet's columns correspond to the
        expressions in the head.
        If head expressions are arguments of a functor, the latter will
        be returned as the second output, defaulted as None.

        Parameters
        ----------
        head : Union[fe.Symbol, Tuple[fe.Expression, ...]]
            see description
        predicate : fe.Expression
            see description
        show_rewritten : bool
            If True, print the Datalog program after magic-sets rewriting.
        dry_run : bool
            If True, print the rewritten program and return immediately
            without running the chase.

        Returns
        -------
        Tuple[AbstractSet, Optional[fe.Symbol]]
            see description

        Examples
        --------
        Note: example ran with pandas backend
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     e.l2[e.x, e.y] = e.l[e.x, e.y] & (e.x == e.y)
        ...     s1 = nl._execute_query(tuple(), e.l2[e.x, e.y])
        ...     s2 = nl._execute_query((e.x,), e.l2[e.x, e.y])
        ...     s3 = nl._execute_query(e.l2[e.x, e.y], e.l2[e.x, e.y])
        >>> s1
        (
            C{
                Empty DataFrame
                Columns: []
                Index: [0]
                : typing.AbstractSet
            },
            None
        )
        >>> s2
        (
            C{
                    x
                0   2
                : typing.AbstractSet
            },
            None
        )
        >>> s3
        (
            C{
                    x   y
                0   2   2
                : typing.AbstractSet
            },
            S{
                l2: Unknown
            }
        )
        """
        functor_orig = None
        if isinstance(head, fe.Operation):
            functor_orig = head.expression.functor

        self.program_ir.symbol_table = self.symbol_table.create_scope()
        try:
            magic_query_expression, reachable_rules = self._magic_sets_rewrite_query(
                head, predicate
            )
            if self._handle_rewritten_output(
                magic_query_expression, show_rewritten, dry_run
            ):
                solution_set = ir.Constant(WrappedRelationalAlgebraFrozenSet())
            else:
                solution = self.chase_class(
                    self.program_ir, rules=reachable_rules
                ).build_chase_solution()

                solution_set = solution.get(
                    magic_query_expression.consequent.functor,
                    ir.Constant(WrappedRelationalAlgebraFrozenSet())
                )

                if isinstance(head, tuple):
                    row_type = solution_set.value.row_type
                    solution_set = NamedRelationalAlgebraFrozenSet(
                        tuple(s.expression.name for s in head),
                        solution_set.value.unwrap()
                    )
                    solution_set.row_type = row_type
        finally:
            self.program_ir.symbol_table = self.symbol_table.enclosing_scope

        return solution_set, functor_orig

    def magic_sets_rewrite_program(self, query_expression):
        goal, mr = magic_rewrite(query_expression.consequent, self.program_ir)
        self.program_ir.walk(mr)
        new_query_expression = self.program_ir.symbol_table[goal].formulas[0]
        return new_query_expression

    def solve_all(self) -> Dict[str, NamedRelationalAlgebraFrozenSet]:
        """
        Returns a dictionary of "predicate_name": "Content"
        for all elements in the solution of the Datalog program.

        Returns
        -------
        Dict[str, NamedRelationalAlgebraFrozenSet]
            extensional and intentional facts that have been derived
            through the current program

        Examples
        --------
        Note: example ran with pandas backend
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (2, 2)], name="l")
        l: typing.AbstractSet[typing.Tuple[int, int]] = [(1, 2), (2, 2)]
        >>> with nl.scope as e:
        ...     e.l2[e.x] = e.l[e.x, e.y] & (e.x == e.y)
        ...     solution = nl.solve_all()
        """
        solution_ir = self.chase_class(self.program_ir).build_chase_solution()

        solution = {}
        for k, v in solution_ir.items():
            solution[predicate_identity(k)] = NamedRelationalAlgebraFrozenSet(
                self.predicate_parameter_names(k),
                v.value.unwrap(),
            )
            solution[predicate_identity(k)].row_type = v.value.row_type
        return solution

    def reset_program(self) -> None:
        """Clears current symbol table"""
        self.symbol_table.clear()

    def add_tuple_set(
        self, iterable: Iterable, type_: Type = Unknown, name: str = None
    ) -> fe.Symbol:
        """
        Creates an AbstractSet fe.Symbol containing the elements specified in
        the iterable with a List[Tuple[Any, ...]] format (see examples).
        Typically used to create extensional facts from existing databases

        Parameters
        ----------
        iterable : Iterable
            typically a list of tuples of values, other formats will
            be interpreted as the latter
        type_ : Type, optional
            type of elements for the tuples, if not specified
            will be inferred from the first element, by default Unknown
        name : str, optional
            name for the AbstractSet symbol, by default None

        Returns
        -------
        fe.Symbol
            see description

        Examples
        --------
        >>> p_ir = DatalogProgram()
        >>> nl = QueryBuilderDatalog(program_ir=p_ir)
        >>> nl.add_tuple_set([(1, 2), (3, 4)], name="l1")
        l1: typing.AbstractSet[typing.Tuple[int, int]] = \
            [(1, 2), (3, 4)]
        >>> nl.add_tuple_set([[1, 2, 3], (3, 4)], name="l2")
        l2: typing.AbstractSet[typing.Tuple[int, int, float]] = \
            [(1, 2, 3.0), (3, 4, nan)]
        >>> nl.add_tuple_set((1, 2, 3), name="l3")
        l3: typing.AbstractSet[typing.Tuple[int]] = \
            [(1,), (2,), (3,)]
        """
        if name is None:
            name = str(uuid1())

        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = ir.Symbol[AbstractSet[type_]](name)
        if isinstance(iterable, pd.DataFrame):
            iterable = iterable.rename(
                columns={n: i for i, n in enumerate(iterable.columns)}
            )
        self.program_ir.add_extensional_predicate_from_tuples(
            symbol, iterable, type_=type_
        )

        return fe.Symbol(self, name)

    def predicate_parameter_names(
        self, predicate_name: Union[str, fe.Symbol, fe.Expression]
    ) -> Tuple[str]:
        """
        Get the names of the parameters for the given predicate

        Parameters
        ----------
        predicate_name : Union[str, fe.Symbol, fe.Expression]
            predicate to obtain the names from

        Returns
        -------
        tuple[str]
            parameter names
        """
        predicate_name = self._get_predicate_name(predicate_name)
        parameter_names = []
        pcount = defaultdict(lambda: 0)
        for s in self.program_ir.predicate_terms(predicate_name):
            param_name = self._obtain_parameter_name(s)
            pcount[param_name] += 1
            if pcount[param_name] > 1:
                param_name = f"{param_name}_{pcount[param_name] - 1}"
            parameter_names.append(param_name)
        return tuple(parameter_names)

    def _obtain_parameter_name(self, parameter_expression):
        if hasattr(parameter_expression, "name"):
            param_name = parameter_expression.name
        elif hasattr(parameter_expression, "functor") and hasattr(
            parameter_expression.functor, "name"
        ):
            param_name = parameter_expression.functor.name
        else:
            param_name = ir.Symbol.fresh().name
        return param_name

    def _get_predicate_name(self, predicate_name):
        if isinstance(predicate_name, ir.Symbol):
            return predicate_name
        elif isinstance(predicate_name, fe.Symbol):
            predicate_name = predicate_name.neurolang_symbol
        elif isinstance(predicate_name, fe.Expression) and isinstance(
            predicate_name.expression, ir.Symbol
        ):
            predicate_name = predicate_name.expression
        elif not isinstance(predicate_name, str):
            raise ValueError(f"{predicate_name} is not a string or symbol")
        return predicate_name
