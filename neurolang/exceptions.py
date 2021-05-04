class NeuroLangException(Exception):
    """Base class for NeuroLang Exceptions"""

    pass


class UnexpectedExpressionError(NeuroLangException):
    pass


class NeuroLangNotImplementedError(NeuroLangException):
    pass


class ForbiddenExpressionError(NeuroLangException):
    pass


class ForbiddenDisjunctionError(ForbiddenExpressionError):
    pass


class ForbiddenExistentialError(ForbiddenExpressionError):
    pass


class RelationalAlgebraError(NeuroLangException):
    pass


class ProjectionOverMissingColumnsError(RelationalAlgebraError):
    pass


class RelationalAlgebraNotImplementedError(
    RelationalAlgebraError, NotImplementedError
):
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
    pass


class ForbiddenRecursivityError(UnsupportedProgramError):
    pass


class ForbiddenUnstratifiedAggregation(UnsupportedProgramError):
    """
    The given Datalog program is not valid for aggregation.

    """
    pass


class WrongArgumentsInPredicateError(NeuroLangException):
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
