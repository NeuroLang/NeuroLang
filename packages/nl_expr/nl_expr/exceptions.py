class NeuroLangException(Exception):
    """Base class for NeuroLang Exceptions"""

    #: Actionable hint shown to users by the error formatter, if any.
    suggestion = None


class UnexpectedExpressionError(NeuroLangException):
    pass


class NeuroLangNotImplementedError(NeuroLangException):
    pass


class ForbiddenExpressionError(NeuroLangException):
    """
    Generic exception specifying an error in the program.
    """
    pass


class ForbiddenDisjunctionError(ForbiddenExpressionError):
    """
    Probabilistic queries do not support disjunctions.

    A probabilistic choice can be added for a predicate symbol by using the
    `add_probabilistic_choice_from_tuples` method of a probabilistic Neurolang
    engine. But you cannot add multiple probabilistic facts or rule for the
    same
    predicate.

    Examples
    --------
    ProbActivation(r, PROB(r)) :- RegionReported(r, s) & SelectedStudy(s)
    ProbActivation(r, PROB(r)) :- ~RegionReported(r, s) & SelectedStudy(s)

    This example program adds a disjunction of probabilistic queries which is
    not allowed.
    """

    suggestion = (
        "A probabilistic predicate can be defined by a single rule or set "
        "of facts: give each probabilistic rule its own head predicate."
    )


class ForbiddenExistentialError(ForbiddenExpressionError):
    pass


class RelationalAlgebraError(NeuroLangException):
    """
    Base class for Relational Algebra provenance exceptions.
    """

    pass


class NotConjunctiveExpression(NeuroLangException):
    """
    This expression is not conjunctive. In this case, an expression is
    conjunctive if it is a conjunction of:

    - Constant
    - A function or predicate of constants
    """

    pass


class NotConjunctiveExpressionNegation(NotConjunctiveExpression):
    """
    This expression is not conjunctive. In this case, an expression is
    conjunctive if it is a conjunction of:

    - Constant
    - A function or predicate of conjunctive arguments
    - A negated predicate of conjunctive arguments
    """

    pass


class NotConjunctiveExpressionNestedPredicates(NotConjunctiveExpression):
    """
    This expression is not conjunctive. In this case, an expression is
    conjunctive if it is a conjunction of:

    - Constant
    - A function or predicate of conjunctive arguments
    - A quantifier of conjunctive arguments

    Note that in this case, negated predicates are not valid (negation and
    aggregation cannot be used in the same rule).

    Examples
    --------
    StudyMatchingRegionSegregationQuery(count(s), r) :-
        RegionReported(r, s) & ~RegionReported(r2, s)
        & RegionLabel(r2) & (r2 != r)

    The above expression is not conjunctive since it uses an aggregate
    function `count` in combination with a negated predicate
    `~RegionReported`.
    """

    suggestion = (
        "Negation and aggregation cannot be used in the same rule: "
        "compute the negated part in a separate rule first."
    )


class ProjectionOverMissingColumnsError(RelationalAlgebraError):
    """
    One of the predicates in the program has wrong arguments.
    See `WrongArgumentsInPredicateError`
    """

    pass


class RelationalAlgebraNotImplementedError(
    RelationalAlgebraError, NotImplementedError
):
    """
    Neurolang was unable to match one of the relational algebra operations
    defined in the program. This is probably due to a malformed query.
    """

    pass


class ForbiddenBuiltinError(ForbiddenExpressionError):
    pass


class NotInFONegE(ForbiddenExpressionError):
    pass


class NeuroLangFrontendException(NeuroLangException):
    pass


class SymbolNotFoundError(NeuroLangException):
    """
    A symbol is being used in a rule without having been previously
    defined.
    """

    suggestion = (
        "Check the spelling, or define the symbol before using it "
        "(e.g. with nl.add_tuple_set)."
    )


class RuleNotFoundError(NeuroLangException):
    pass


class UnsupportedProgramError(NeuroLangException):
    """
    Some parts of the datalog program are (currently) unsupported.
    """
    pass


class UnsupportedQueryError(NeuroLangException):
    """
    Queries on probabilistic predicates are unsupported.
    """
    pass


class UnsupportedSolverError(NeuroLangException):
    pass


class ProtectedKeywordError(NeuroLangException):
    """
    One of the predicates in the program uses a reserved keyword.
    Reserved keywords include : {PROB, with, exists}
    """
    suggestion = (
        "Rename the predicate: PROB, with and exists are reserved keywords."
    )


class ForbiddenRecursivityError(UnsupportedProgramError):
    """
    The given program cannot be stratified due to recursivity.

    When using probabilistic queries, a query can
    be solved through stratification if the probabilistic and deterministic
    parts are well separated. In case there exists one within-language
    probabilistic query dependency, no probabilistic predicate should appear
    in the stratum that depends on the query.
    The same holds for aggregate or negated queries. If a rule contains an
    aggregate or negated term, all the predicates in the body of the rule
    must be computed in a previous stratum.

    Examples
    --------
    B(x) :- A(x), C(x)
    A(x) :- B(x)

    This program cannot be stratified because it contains a loop in
    the dependencies of each rule. Rule `B(x) :- A(x), C(x)` depends
    on the second rule through its occurrence of the predicate `A(x)`.
    But rule `A(x) :- B(x)` in turn depends on the first rule through
    the `B(x)` predicate.
    """

    suggestion = (
        "Break the dependency cycle: predicates used under negation, "
        "aggregation or probabilistic queries must not depend on the head "
        "of the rule using them."
    )


class ForbiddenUnstratifiedAggregation(UnsupportedProgramError):
    """
    The given Datalog program is not valid for aggregation. Support for
    aggregation is done according to section 2.4.1 of [1]_.

    A program is valid for aggregation if it can be stratified into
    strata P1, . . . , Pn such that, if A :- ...,B,... is a rule in P such
    that A contains an aggregate term, and A is in stratum Pi while B is in
    stratum Pj, **then i > j**.

    In other terms, all the predicates in the body of a rule containing an
    aggregate function must be computed in a previous stratum. Recursion
    through aggregation is therefore not allowed in the same stratum.

    Examples
    --------
    The following datalog program is invalid for stratified aggregation
    p(X) :- q(X).
    p(sum<X>) :- p(X).


    .. [1] T. J. Green, S. S. Huang, B. T. Loo, W. Zhou,
       Datalog and Recursive Query Processing.
       FNT in Databases. 5, 105–195 (2012).
    """

    suggestion = (
        "The predicates in the body of an aggregation rule must not depend "
        "on its head (no recursion through aggregation)."
    )


class WrongArgumentsInPredicateError(NeuroLangException):
    """
    One of the predicates in the query has the wrong number of arguments.

    Examples
    --------
    NetworkReported is defined with two variables but used with three
    in the second rule:

    .. code-block:: text

        NetworkReported(n, s) :- RegionReported(
            r, s
        ) & RegionInNetwork(r, n)
        StudyMatchingNetworkQuery(s, n) :- (
            RegionReported("VWFA", s)
            & NetworkReported(n, s, r)
        )
    """

    suggestion = (
        "Use every predicate with the same number of arguments as in its "
        "definition."
    )


class PredicateArityError(WrongArgumentsInPredicateError):
    """
    A predicate is used with a number of arguments other than its columns.

    Parameters
    ----------
    predicate : str
        Name of the predicate.
    arity : int
        Number of columns of the predicate.
    used : int
        Number of arguments the predicate is used with.

    Notes
    -----
    The message only states the counts. When a single argument is used for
    a predicate with several columns, as a SQUALL noun without ``in ND``
    does, `suggestion` tells how to use all its columns; when more
    arguments than columns are used, as a SQUALL verb with too many
    objects does, it points to a tuple subject on the relation storing
    the extra value.
    """

    def __init__(self, predicate, arity, used):
        """Build the message from the predicate name and the counts."""
        columns = "column" if arity == 1 else "columns"
        super().__init__(
            f"`{predicate}` has {arity} {columns} but is used with {used}"
        )
        self.predicate = predicate
        self.arity = arity
        self.used = used
        if used == 1 and arity > 1:
            self.suggestion = (
                f"If `{predicate}` is a noun, use `in {arity}D` or a tuple "
                f"of {arity} variables such as `("
                + "; ".join(f"?x{i}" for i in range(1, arity + 1))
                + ")`."
            )
        elif used > arity:
            self.suggestion = (
                f"If `{predicate}` is a verb with several objects, as in "
                "`?s weighs ?t with ?w`, its relation needs one column per "
                "argument. To use a value stored in another relation, "
                "name that relation with a tuple subject, e.g. "
                "`where (?s; ?t; ?w) is a Term_in_study_tfidf`."
            )


class TranslateToNamedRAException(NeuroLangException):
    """
    Base exception for errors translating Datalog to Named Relational Algebra.
    """

    pass


class NoValidChaseClassForStratumException(NeuroLangException):
    """
    Neurolang implements stratified datalog which splits a datalog program
    into several independent strata that can each be solved by a specific
    chase algorithm based on the properties of the rules in the stratum
    (using negation, aggregation and/or recursion).

    This exception is raised if there is no valid algorithm available to
    solve a specific stratum; e.g. no recursive compatible algorithm was
    provided to solve a recursive stratum.

    See `neurolang.datalog.chase.__init__.py` for available chase
    implementations.
    """

    pass


class CouldNotTranslateConjunctionException(TranslateToNamedRAException):
    """
    This conjunctive formula could not be translated into an equivalent
    relational algebra representation. This is probably because the
    formula is not in *modified relational algebra normal form*.

    Generally speaking, the formula must be expressed in *conjunctive normal
    form* (CNF) or *disjunctive normal form* (DNF): as either a conjunction of
    disjunctions or disjunction of conjunctions.

    A conjunction with variables that are not bound by any of its positive
    atoms raises :class:`UnboundVariablesError` instead.

    See 5.4.7 from [1]_.

    Examples
    --------
    .. code-block:: text

        PositiveReverseInferenceSegregationQuery(
            t, n, PROB(t, n)
        ) :- (TopicAssociation(t, s) // SelectedStudy(s)) // (
            StudyMatchingNetworkQuery(s, n) & SelectedStudy(s)
        )

    This formula is not in DNF since it is a disjunction of a disjunction
    (TopicAssociation(t, s) // SelectedStudy(s)) and a conjunction
    (StudyMatchingNetworkQuery(s, n) & SelectedStudy(s)).
    A valid query would be:

    .. code-block:: text

        PositiveReverseInferenceSegregationQuery(
            t, n, PROB(t, n)
        ) :- (TopicAssociation(t, s) & SelectedStudy(s)) // (
            StudyMatchingNetworkQuery(s, n) & SelectedStudy(s)
        )

    .. [1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
        (Addison Wesley, 1995), Addison-Wesley.
    """

    suggestion = (
        "Rewrite the formula in conjunctive or disjunctive normal form "
        "(CNF or DNF)."
    )

    def __init__(self, output):
        super().__init__(f"Could not translate conjunction: {output}")
        self.output = output


class NegativeFormulaNotSafeRangeException(TranslateToNamedRAException):
    """
    This rule is not *range restricted* and cannot be solved in
    *nonrecursive datalog with negation*. One of the variables in this rule
    appears in a negated literal without also appearing in a non-negated
    literal.

    A datalog rule composed of literals of the form R(v) or ¬R(v) is
    *range restricted* if each variable x occurring in the rule occurs in at
    least one literal of the form R(v) (non-negated literal) in the rule body.
    See 5.2 from [1]_.

    Examples
    --------
    .. code-block:: text

        StudyNotMatchingSegregationQuery(s, n) :- (
            ~StudyMatchingNetworkQuery(s, n)
            & Network(n)
        )

    Variable ``s`` is present in the negated ``StudyMatchingNetworkQuery``
    literal but is not present in a non-negated literal. A valid query body
    would be:

    .. code-block:: text

        StudyNotMatchingSegregationQuery(s, n) :- (
            ~StudyMatchingNetworkQuery(s, n)
            & Study(s) & Network(n)
        )

    .. [1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
        (Addison Wesley, 1995), Addison-Wesley.
    """

    suggestion = (
        "Every variable of a negated predicate must also appear in a "
        "non-negated predicate of the same rule body."
    )

    def __init__(self, formula):
        super().__init__(f"Negative predicate {formula} is not safe range")
        self.formula = formula


class UnboundVariablesError(ForbiddenExpressionError):
    """
    This rule is not *range restricted*: some of its variables do not
    appear in any positive atom of its body.

    Every variable of a rule must occur in a non-negated predicate of the
    body, or be defined by an equality ``x == expression`` whose
    expression only uses such variables. Functions, comparisons and
    negated predicates can only filter the values of bound variables,
    they cannot enumerate them. See 5.2 from [1]_.

    Examples
    --------
    .. code-block:: text

        reports_close(i, j, k, s) :- (
            (euclidean(i, j, k, i1, j1, k1) < 2.)
            & voxel(i, j, k) & peak_reported(i, j, k, s)
        )

    Variables ``i1``, ``j1`` and ``k1`` appear only as arguments of the
    function ``euclidean``. A valid rule binds them with a predicate:

    .. code-block:: text

        reports_close(i, j, k, s) :- (
            (euclidean(i, j, k, i1, j1, k1) < 2.)
            & voxel(i, j, k) & peak_reported(i1, j1, k1, s)
        )

    Parameters
    ----------
    rule : str or None
        Name of the head predicate of the rule, None for a query or when
        the rule is unknown.
    variables : tuple of str
        Names of the unbound variables.
    occurrences : tuple of str
        The formulas of the body where the unbound variables appear.
    example : str, optional
        A predicate that would bind the variables in the body.
    query : bool
        Whether the body is the body of a query.

    .. [1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
        (Addison Wesley, 1995), Addison-Wesley.
    """

    def __init__(
        self, rule, variables, occurrences, example=None, query=False
    ):
        self.rule = rule
        self.variables = tuple(variables)
        self.occurrences = tuple(occurrences)
        self.example = example
        self.query = query
        super().__init__(self._message())
        self.suggestion = self._suggestion()

    def __reduce__(self):
        return type(self), (
            self.rule, self.variables, self.occurrences, self.example,
            self.query,
        )

    def located(self, rule=None, query=False):
        """
        Return this error, reported for the rule named *rule*, or for a
        query.
        """
        return type(self)(
            rule, self.variables, self.occurrences, self.example, query
        )

    def _names(self):
        return ", ".join(self.variables)

    def _message(self):
        if self.rule is not None:
            where = self.rule
        else:
            where = "the query" if self.query else "a rule body"
        plural = len(self.variables) > 1
        return (
            f"In {where}: the variable{'s' if plural else ''} "
            f"{self._names()} {'appear' if plural else 'appears'} only in "
            + " and ".join(f"`{o}`" for o in self.occurrences)
            + "; every variable must appear in a positive atom of the "
            "rule body, or be defined by an equality `x == expression` "
            "over such variables"
        )

    def _suggestion(self):
        tip = f"Add a predicate over {self._names()} to the rule body"
        if self.example:
            return f"{tip}, such as {self.example}."
        return (
            f"{tip}, or define each with an equality such as "
            f"`{self.variables[0]} == <expression over bound variables>`."
        )


class NegativeFormulaNotNamedRelationException(TranslateToNamedRAException):
    """
    This rule contains a negative literal R(v) which was not previously
    defined as a non-negated relation.

    Examples
    --------
    t(x, y) :- r(x, y) & q(y, z)
    s(x, y, prob(x, y)) :- ~t(x, x) & q(x, y)
    """

    suggestion = (
        "Negate only named predicates: define the negated formula as its "
        "own rule first."
    )

    def __init__(self, formula):
        super().__init__(f"Negative formula {formula} is not a named relation")
        self.formula = formula


class NonLiftableException(UnsupportedSolverError):
    pass


class NotRankedException(NonLiftableException):
    pass


class NotUnateException(NonLiftableException):
    pass

class InvalidCommandExpression(ForbiddenExpressionError):
    """
    Invalid Command statement.
    """

    pass


class ParserError(NeuroLangException):
    def __init__(self, message, line=None, column=None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self):
        return self.message


class UnexpectedTokenError(ParserError):
    pass


class UnexpectedCharactersError(ParserError):
    pass


class SquallSemanticError(NeuroLangException):

    """Semantic error in a SQUALL program — parse succeeded but semantics invalid.

    Raised when the SQUALL transformer encounters a valid parse tree that
    represents a logically invalid or unsupported construction, such as:
    - a rule with an empty body
    - an anaphoric ``the Noun`` with no matching ``for every Noun`` in scope
    - an unrecognised comparison or aggregation function
    - a missing determiner or noun group in a rule body

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    line : int, optional
        1-based source line number where the error was detected.
    column : int, optional
        1-based source column number where the error was detected.
    source_line : str, optional
        The full source line text for context display.

    """

    def __init__(self, message, *, line=None, column=None, source_line=None):
        """Initialise with message and optional source location."""
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.source_line = source_line

    def __str__(self):
        if self.source_line and self.column is not None:
            pointer = " " * (max(0, self.column - 1)) + "^~~~~~"
            return (
                f"SQUALL semantic error (line {self.line}):\n"
                f"    {self.source_line.strip()}\n"
                f"    {pointer}\n"
                f"{self.message}"
            )
        if self.line is not None:
            return (
                f"SQUALL semantic error (line {self.line}, "
                f"col {self.column}): {self.message}"
            )
        return f"SQUALL semantic error: {self.message}"