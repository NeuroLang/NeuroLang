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
    pass


class ForbiddenExistentialError(ForbiddenExpressionError):
    pass


class RelationalAlgebraError(NeuroLangException):
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
    defined in the program. This is probably due to an malformed query.
    """
    pass


class ForbiddenBuiltinError(ForbiddenExpressionError):
    pass


class NeuroLangFrontendException(NeuroLangException):
    pass


class SymbolNotFoundError(NeuroLangException):
    pass


class RuleNotFoundError(NeuroLangException):
    pass


class UnsupportedProgramError(NeuroLangException):
    """
    Some parts of the datalog program are (currently) unsupported.
    """
    pass


class UnsupportedQueryError(NeuroLangException):
    pass


class UnsupportedSolverError(NeuroLangException):
    pass


class ProtectedKeywordError(NeuroLangException):
    """
    One of the predicates in the program uses a reserved keyword.
    Reserved keywords include : {PROB}
    """
    pass


class ForbiddenRecursivityError(UnsupportedProgramError):
    """
    The given program cannot be stratified due to recursivity. A query can
    be solved through stratification if the probabilistic and deterministic
    parts are well separated. In case there exists one within-language
    probabilistic query dependency, no probabilistic predicate should appear
    in the stratum that depends on the query.

    Examples
    --------
    B(x) :- A(x), C(x),
    A(x) :- B(x)
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

    e.NetworkReported[e.n, e.s] = e.RegionReported(
        e.r, e.s
    ) & e.RegionInNetwork(e.r, e.n)
    e.StudyMatchingNetworkQuery[e.s, e.n] = (
        e.RegionReported("VWFA", e.s)
        & e.NetworkReported(e.n, e.s, e.r)
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
    named relational algebra representation. This is probably because the
    formula is not in *modified relational algebra normal form*.

    See 5.4.7 from [1]_.

    Examples
    --------
    e.PositiveReverseInferenceSegregationQuery[
        e.t, e.n, e.PROB(e.t, e.n)
    ] = (e.TopicAssociation(e.t, e.s) // e.SelectedStudy(e.s)) // (
        e.StudyMatchingNetworkQuery(e.s, e.n) & e.SelectedStudy(e.s)
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
    e.StudyNotMatchingSegregationQuery[e.s, e.n] = (
        ~e.StudyMatchingNetworkQuery(e.s, e.n)
        & e.Network(e.n)
    )

    Variable `e.s` is present in the negated `e.StudyMatchingNetworkQuery`
    literal but is not present in a non-negated literal. A valid query body
    would be :

    e.StudyNotMatchingSegregationQuery[e.s, e.n] = (
        ~e.StudyMatchingNetworkQuery(e.s, e.n)
        & e.Study(e.s) & e.Network(e.n)
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
    t[x, y] = r(x, y) & q(y, z)
    s[x, y, prob(x, y)] = ~t(x, x) & q(x, y)
    """
    def __init__(self, formula):
        super().__init__(f"Negative formula {formula} is not a named relation")
        self.formula = formula
