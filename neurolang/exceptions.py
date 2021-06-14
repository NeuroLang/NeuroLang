class NeuroLangException(Exception):
    """Base class for NeuroLang Exceptions"""

    pass


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
    ----------
    ProbActivation(r, PROB(r)) :- RegionReported(r, s) & SelectedStudy(s)
    ProbActivation(r, PROB(r)) :- ~RegionReported(r, s) & SelectedStudy(s)

    This example program adds a disjunction of probabilistic queries which is
    not allowed.
    """

    pass


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
    conjunctive if it is a conjunction of
      - Constant
      - A function or predicate of constants
    """

    pass


class NotConjunctiveExpressionNegation(NotConjunctiveExpression):
    """
    This expression is not conjunctive. In this case, an expression is
    conjunctive if it is a conjunction of
      - Constant
      - A function or predicate of conjunctive arguments
      - A negated predicate of conjunctive arguments
    """

    pass


class NotConjunctiveExpressionNestedPredicates(NotConjunctiveExpression):
    """
    This expression is not conjunctive. In this case, an expression is
    conjunctive if it is a conjunction of
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

    pass


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


class NeuroLangFrontendException(NeuroLangException):
    pass


class SymbolNotFoundError(NeuroLangException):
    """
    A symbol is being used in a rule without having been previously
    defined.
    """

    pass


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

    Examples
    ----------
    NeuroLangException : [type]
        [description]
    """
    pass


class UnsupportedSolverError(NeuroLangException):
    pass


class ProtectedKeywordError(NeuroLangException):
    """
    One of the predicates in the program uses a reserved keyword.
    Reserved keywords include : {PROB, with, exists}
    """
    pass


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
    on the second rule through its occurence of the predicate `A(x)`.
    But rule `A(x) :- B(x)` in turn depends on the first rule through
    the `B(x)` predicate.
    """

    pass


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

    pass


class WrongArgumentsInPredicateError(NeuroLangException):
    """
    One of the predicates in the query has the wrong number of arguments.

    Examples
    --------
    NetworkReported is defined with two variables but used with three
    in the second rule:

    NetworkReported(e.n, e.s) :- RegionReported(
        r, s
    ) & RegionInNetwork(r, n)
    StudyMatchingNetworkQuery(s, n) :- (
        RegionReported("VWFA", s)
        & NetworkReported(e.n, e.s, e.r)
    )
    """

    pass


class TranslateToNamedRAException(NeuroLangException):
    """
    Base class for `tranlate_to_named_ra.py`.
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

    Generaly speaking, the formula must be expressed in *conjunctive normal
    form* (CNF) or *disjunctive normal form* (DNF): as either a conjunction of
    disjunctions or disjunction of conjunctions.

    See 5.4.7 from [1]_.

    Examples
    --------
    PositiveReverseInferenceSegregationQuery(
        t, n, PROB(t, n)
    ) :- (TopicAssociation(t, s) // SelectedStudy(s)) // (
        StudyMatchingNetworkQuery(s, n) & SelectedStudy(s)
    )

    This formula is not in DNF since it is a disjunction of a disjunction
    (TopicAssociation(t, s) // SelectedStudy(s)) and a conjunction 
    (StudyMatchingNetworkQuery(s, n) & SelectedStudy(s)).
    A valid query would be :

    PositiveReverseInferenceSegregationQuery(
        t, n, PROB(t, n)
    ) :- (TopicAssociation(t, s) & SelectedStudy(s)) // (
        StudyMatchingNetworkQuery(s, n) & SelectedStudy(s)
    )

    .. [1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
        (Addison Wesley, 1995), Addison-Wesley.
    """

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
    StudyNotMatchingSegregationQuery(s, n) :- (
        ~StudyMatchingNetworkQuery(s, n)
        & Network(n)
    )

    Variable `s` is present in the negated `StudyMatchingNetworkQuery`
    literal but is not present in a non-negated literal. A valid query body
    would be :

    StudyNotMatchingSegregationQuery(s, n) :- (
        ~StudyMatchingNetworkQuery(s, n)
        & Study(s) & Network(n)
    )

    .. [1] S. Abiteboul, R. Hull, V. Vianu, Foundations of databases
        (Addison Wesley, 1995), Addison-Wesley.
    """

    def __init__(self, formula):
        super().__init__(f"Negative predicate {formula} is not safe range")
        self.formula = formula


class NegativeFormulaNotNamedRelationException(TranslateToNamedRAException):
    """
    This rule contains a negative literal R(v) which was not previously
    defined as a non-negated relation.

    Examples
    --------
    t(x, y) :- r(x, y) & q(y, z)
    s(x, y, prob(x, y)) :- ~t(x, x) & q(x, y)
    """

    def __init__(self, formula):
        super().__init__(f"Negative formula {formula} is not a named relation")
        self.formula = formula
