import io
import itertools
from typing import AbstractSet, Callable, Tuple

import numpy as np
import pandas as pd
import pytest

from ...exceptions import (
    NegativeFormulaNotNamedRelationException,
    NegativeFormulaNotSafeRangeException,
    UnsupportedProgramError,
    UnsupportedQueryError,
    UnsupportedSolverError,
)
from ...probabilistic import dalvi_suciu_lift
from ...probabilistic.exceptions import (
    ForbiddenConditionalQueryNonConjunctive,
    RepeatedTuplesInProbabilisticRelationError,
    UnsupportedProbabilisticQueryError,
)
from ...regions import SphericalVolume
from ...utils.relational_algebra_set import (
    NamedRelationalAlgebraFrozenSet,
    RelationalAlgebraFrozenSet,
)
from ..probabilistic_frontend import NeurolangPDL


def assert_almost_equal(set_a, set_b):
    for tupl_a in set_a:
        assert any(
            all(
                np.isclose(term_a, term_b)
                if isinstance(term_a, (float, np.float32, np.float64))
                else term_a == term_b
                for term_a, term_b in zip(tupl_a, tupl_b)
            )
            for tupl_b in set_b
        )


def test_add_uniform_probabilistic_choice_set():
    nl = NeurolangPDL()

    prob = [(a,) for a in range(10)]
    prob_set = nl.add_uniform_probabilistic_choice_over_set(prob, "prob")
    res = nl[prob_set]

    assert prob_set.type is AbstractSet[Tuple[float, int]]
    assert res.type is AbstractSet[Tuple[float, int]]
    assert res.value == frozenset((1 / len(prob), a) for a in range(10))

    d = [("a",), ("b",), ("c",), ("d",)]
    data = nl.add_uniform_probabilistic_choice_over_set(d, name="data")
    res_d = nl[data]

    assert data.type is AbstractSet[Tuple[float, str]]
    assert res_d.type is AbstractSet[Tuple[float, str]]

    d = pd.DataFrame([("a",), ("b",), ("c",), ("d",)], columns=["ids"])
    data2 = nl.add_uniform_probabilistic_choice_over_set(d, name="data2")
    res_d2 = nl[data]

    assert data2.type is AbstractSet[Tuple[float, str]]
    assert res_d2.type is AbstractSet[Tuple[float, str]]


def test_deterministic_query():
    nl = NeurolangPDL()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_tuple_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_tuple_set(d2, name="data2")

    with nl.scope as e:
        e.query1[e.y] = data1[e.x] & data2[e.x, e.y]
        res = nl.solve_all()

    assert "query1" in res.keys()
    q1 = res["query1"].as_pandas_dataframe().values
    assert len(q1) == 4
    for elem in q1:
        assert elem[0] in ["a", "b", "c", "d"]


def test_probabilistic_query():
    nl = NeurolangPDL()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_uniform_probabilistic_choice_over_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_uniform_probabilistic_choice_over_set(d2, name="data2")

    d3 = [("a",), ("b",), ("c",)]
    data3 = nl.add_uniform_probabilistic_choice_over_set(d3, name="data3")

    with nl.scope as e:
        e.query1[e.y, e.PROB[e.y]] = data1[e.x] & data2[e.x, e.y]
        e.query2[e.y, e.PROB[e.y]] = e.query1[e.y] & data3[e.y]
        with pytest.raises(UnsupportedProgramError):
            nl.solve_all()


def test_marg_query():
    nl = NeurolangPDL()
    nl.add_probabilistic_choice_from_tuples(
        {(0.2, "a"), (0.3, "b"), (0.5, "c")}, name="P"
    )
    nl.add_tuple_set({(1, "a"), (2, "a"), (2, "b")}, name="Q")
    nl.add_tuple_set({(1, "a"), (1, "b"), (2, "b"), (2, "c")}, name="R")

    with nl.scope as e:
        e.Z[e.x, e.z, e.PROB[e.x, e.z]] = (e.Q(e.x, e.y) & e.P(e.y)) // (
            e.R(e.z, e.y) & e.P(e.y)
        )

        res = nl.solve_all()

    expected = RelationalAlgebraFrozenSet(
        {
            (1, 1, 0.4),
            (2, 1, 1.0),
            (2, 2, 0.375),
        }
    )
    assert_almost_equal(res["Z"], expected)


def test_mixed_queries():
    nl = NeurolangPDL()
    d1 = [(1,), (2,), (3,), (4,), (5,)]
    data1 = nl.add_tuple_set(d1, name="data1")

    d2 = [(2, "a"), (3, "b"), (4, "d"), (5, "c"), (7, "z")]
    data2 = nl.add_tuple_set(d2, name="data2")

    d3 = [("a",), ("b",), ("c",), ("d",)]
    data3 = nl.add_uniform_probabilistic_choice_over_set(d3, name="data3")

    with nl.scope as e:
        e.tmp[e.x, e.y] = data1[e.x] & data2[e.x, e.y]
        e.query1[e.x, e.y, e.PROB[e.x, e.y]] = e.tmp[e.x, e.y]
        e.query2[e.y, e.PROB[e.y]] = e.tmp[e.x, e.y] & data3[e.y]
        res = nl.solve_all()

    assert "query1" in res.keys()
    assert "query2" in res.keys()
    assert len(res["query2"].columns) == 2
    q2 = res["query2"].as_pandas_dataframe().values
    assert len(q2) == 4
    for elem in q2:
        assert elem[1] == 0.25
        assert elem[0] in ["a", "b", "c", "d"]


def test_ontology_query():

    test_case = """
    <rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:first="http://www.w3.org/2002/03owlt/hasValue/premises001#"
        xml:base="http://www.w3.org/2002/03owlt/hasValue/premises001" >
        <owl:Ontology/>
        <owl:Class rdf:ID="r">
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="#p2"/>
                <owl:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:hasValue>
            </owl:Restriction>
        </rdfs:subClassOf>
        </owl:Class>
        <owl:ObjectProperty rdf:ID="p"/>
        <owl:ObjectProperty rdf:ID="p2"/>
        <owl:Class rdf:ID="c"/>
        <first:r rdf:ID="i">
        <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Thing"/>
        <first:p>
            <owl:Thing rdf:ID="o" />
        </first:p>
        </first:r>
    </rdf:RDF>
    """

    nl = NeurolangPDL()
    nl.load_ontology(io.StringIO(test_case))

    p2 = nl.new_symbol(
        name="http://www.w3.org/2002/03owlt/hasValue/premises001#p2"
    )

    with nl.scope as e:
        e.answer[e.x, e.y] = p2[e.x, e.y]
        solution_instance = nl.solve_all()

    resp = list(solution_instance["answer"].itervalues())
    assert (
        "http://www.w3.org/2002/03owlt/hasValue/premises001#i",
        "true",
    ) in resp


def test_simple_within_language_succ_query():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("e",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        res = nl.solve_all()
    assert "Z" in res.keys()
    df = res["Z"].as_pandas_dataframe()
    assert len(df) == 1
    assert tuple(df.values[0])[0] == "a"
    assert np.isclose(tuple(df.values[0])[1], 1 / 9)


def test_within_language_succ_query():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [
            ("a", "b"),
            ("b", "c"),
            ("b", "d"),
        ],
        name="P",
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x, e.y] & Q[e.x]
        res = nl.solve_all()
    assert "Z" in res.keys()
    df = res["Z"].as_pandas_dataframe()
    assert np.isclose(df.loc[df.iloc[:, 0] == "b"].iloc[0, 1], 2 / 3 / 2)
    assert np.isclose(df.loc[df.iloc[:, 0] == "a"].iloc[0, 1], 1 / 3 / 2)


def test_solve_query():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        res = nl.query((e.x, e.p), e.Z[e.x, e.p])
        res_2 = nl.query(e.p, e.Z(e.x, e.p))
    expected = RelationalAlgebraFrozenSet(
        [
            ("a", 1 / 9),
            ("c", 1 / 9),
        ]
    )
    assert_almost_equal(res, expected)
    assert_almost_equal(res_2, expected.projection(1))


def test_solve_query_prob_col_not_last():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.PROB[e.x], e.x] = P[e.x] & Q[e.x]
        res = nl.query((e.p, e.x), e.Z[e.p, e.x])
    expected = RelationalAlgebraFrozenSet(
        [
            (1 / 9, "a"),
            (1 / 9, "c"),
        ]
    )
    assert_almost_equal(res, expected)


def test_solve_boolean_query():
    nl = NeurolangPDL()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with pytest.raises(UnsupportedProbabilisticQueryError):
        with nl.scope as e:
            e.ans[e.PROB()] = P[e.x] & Q[e.x]
            nl.query((e.p), e.ans[e.p])


def test_solve_complex_stratified_query():
    """
    R(1, 2) : 0.3
    R(1, 4) : 0.7
    R(2, 2) : 0.2
    R(2, 4) : 0.6
    R(2, 6) : 0.8
    Q(4) : 0.2 v Q(6) : 0.8
    A(x, PROB[x])       :- Q(x), R(1, x), R(2, x)
    B(x, y, PROB[x, y]) :- Q(y), R(1, x), R(2, x)
    C(x, y, p1, p2)     :- A(x, p1), B(x, y, p2)

    """
    nl = NeurolangPDL()
    R = nl.add_probabilistic_facts_from_tuples(
        [(0.3, 1, 2), (0.7, 1, 4), (0.2, 2, 2), (0.6, 2, 4), (0.8, 2, 6)],
        name="R",
    )
    Q = nl.add_probabilistic_choice_from_tuples([(0.2, 4), (0.8, 6)], name="Q")
    with nl.scope as e:
        e.A[e.x, e.PROB[e.x]] = Q[e.x] & R[1, e.x] & R[2, e.x]
        e.B[e.x, e.y, e.PROB[e.x, e.y]] = Q[e.y] & R[1, e.x] & R[2, e.x]
        e.C[e.x, e.y, e.p1, e.p2] = e.A[e.x, e.p1] & e.B[e.x, e.y, e.p2]
        res = nl.query((e.x, e.y, e.h, e.z), e.C[e.x, e.y, e.h, e.z])
    expected = RelationalAlgebraFrozenSet(
        [
            (4, 4, 0.2 * 0.7 * 0.6, 0.2 * 0.7 * 0.6),
            (4, 6, 0.2 * 0.7 * 0.6, 0.8 * 0.7 * 0.6),
        ]
    )
    assert_almost_equal(res, expected)


def test_solve_complex_stratified_query_with_deterministic_part():
    nl = NeurolangPDL()
    A = nl.add_tuple_set([("a",), ("b",), ("c",)], name="A")
    B = nl.add_tuple_set([("a",), ("b",)], name="B")
    P = nl.add_probabilistic_facts_from_tuples(
        [(0.2, "a"), (0.8, "b")],
        name="P",
    )
    with nl.scope as e:
        e.C[e.x, e.y] = A[e.x] & A[e.y] & B[e.x]
        e.D[e.x, e.y, e.PROB[e.x, e.y]] = e.C[e.x, e.y] & P[e.y]
        res = nl.query((e.x, e.y, e.p), e.D[e.x, e.y, e.p])
    expected = RelationalAlgebraFrozenSet(
        [
            ("a", "a", 0.2),
            ("a", "b", 0.8),
            ("b", "a", 0.2),
            ("b", "b", 0.8),
        ]
    )
    assert_almost_equal(res, expected)


def test_neurolange_dl_deterministic_negation():
    neurolang = NeurolangPDL()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name="q")
    s[x, y] = ~q(x, x) & q(x, y)

    res = neurolang.solve_all()

    assert res["s"].to_unnamed() == {(i, j) for i, j in dataset if i != j}


@pytest.mark.skip
def test_neurolange_dl_probabilistic_negation():
    neurolang = NeurolangPDL()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")
    prob = neurolang.new_symbol(name="PROB")

    dataset_det = {(i, i * 2) for i in range(10)}
    dataset = {((1 + i) / 10, i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset_det, name="q")
    r = neurolang.add_probabilistic_facts_from_tuples(dataset, name="r")

    s[x, y, prob(x, y)] = ~r(x, y) & q(x, y)

    result = neurolang.solve_all()["s"].to_unnamed()
    expected = {(i, j, 1 - p) for (p, i, j) in dataset}
    assert_almost_equal(result, expected)

@pytest.mark.skip
def test_neurolange_dl_probabilistic_negation_not_safe():
    neurolang = NeurolangPDL()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")
    z = neurolang.new_symbol(name="z")
    prob = neurolang.new_symbol(name="PROB")

    dataset_det = {(i, i * 2) for i in range(10)}
    dataset = {((1 + i) / 10, i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset_det, name="q")
    r = neurolang.add_probabilistic_facts_from_tuples(dataset, name="r")

    s[x, y, prob(x, y)] = ~r(x, z) & q(x, y)

    with pytest.raises(NegativeFormulaNotSafeRangeException):
        neurolang.solve_all()


@pytest.mark.skip
def test_neurolange_dl_probabilistic_negation_rule():
    neurolang = NeurolangPDL()
    s = neurolang.new_symbol(name="s")
    t = neurolang.new_symbol(name="t")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")
    z = neurolang.new_symbol(name="z")
    prob = neurolang.new_symbol(name="PROB")

    dataset_det = {(i, i * 2) for i in range(10)}
    dataset = {((1 + i) / 10, i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset_det, name="q")
    r = neurolang.add_probabilistic_facts_from_tuples(dataset, name="r")

    t[x, y] = r(x, y) & q(y, z)
    s[x, y, prob(x, y)] = ~t(x, x) & q(x, y)

    with pytest.raises(NegativeFormulaNotNamedRelationException):
        neurolang.solve_all()


def test_neurolang_dl_aggregation():
    neurolang = NeurolangPDL()
    q = neurolang.new_symbol(name="q")
    p = neurolang.new_symbol(name="p")
    r = neurolang.new_symbol(name="r")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    @neurolang.add_symbol
    def sum_(x):
        return sum(x)

    for i in range(10):
        q[i % 2, i] = True

    p[x, sum_(y)] = q[x, y]

    sol = neurolang.query(r(x, y), p(x, y))

    res_q = {(0, 2 + 4 + 8), (1, 1 + 5 + 9)}

    assert len(sol) == 2
    assert sol[r] == res_q
    assert sol[p] == res_q


def test_post_probabilistic_aggregation():
    nl = NeurolangPDL()
    A = nl.add_probabilistic_facts_from_tuples(
        [(0.2, "a"), (0.9, "b"), (0.5, "c")],
        name="A",
    )
    B = nl.add_probabilistic_choice_from_tuples(
        [(0.2, "a", "c"), (0.7, "b", "c"), (0.1, "a", "d")],
        name="B",
    )

    @nl.add_symbol
    def mysum(x):
        return sum(x)

    with nl.scope as e:
        e.C[e.x, e.y, e.PROB[e.x, e.y]] = A[e.x] & B[e.x, e.y]
        e.D[e.x, e.mysum(e.p)] = e.C[e.x, e.y, e.p]
        res = nl.query((e.x, e.s), e.D[e.x, e.s])

    assert len(res) == 2
    assert res.to_unnamed() == {("a", 0.2 * 0.2 + 0.2 * 0.1), ("b", 0.9 * 0.7)}


def test_empty_result_query():
    nl = NeurolangPDL()
    A = nl.add_tuple_set([("f",), ("d",)], name="A")
    B = nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "a"),
            (0.7, "b"),
            (0.6, "c"),
        ],
        name="B",
    )
    with nl.scope as e:
        e.Q[e.x, e.PROB[e.x]] = A[e.x] & B[e.x]
        res = nl.query((e.x, e.p), e.Q[e.x, e.p])
    assert res.is_empty()
    with nl.scope as e:
        e.Q[e.x] = A[e.x] & A["b"]
        res = nl.query((e.x,), e.Q[e.x])
    assert res.is_empty()


def test_forbidden_query_on_probabilistic_predicate():
    nl = NeurolangPDL()
    A = nl.add_tuple_set([("f",), ("d",)], name="A")
    B = nl.add_probabilistic_facts_from_tuples(
        [(0.2, "a"), (0.7, "b"), (0.6, "c")], name="B"
    )
    with pytest.raises(UnsupportedQueryError):
        with nl.scope as e:
            e.Q[e.x] = A[e.x] & B[e.x] & A["b"]
            nl.query((e.x,), e.Q[e.x])


def test_empty_boolean_query_result():
    nl = NeurolangPDL()
    A = nl.add_tuple_set([("a",), ("b",), ("c",)], name="A")
    B = nl.add_probabilistic_facts_from_tuples(
        [(0.4, "a"), (0.5, "b")], name="B"
    )
    C = nl.add_probabilistic_choice_from_tuples(
        [(0.8, "c"), (0.2, "f")], name="C"
    )
    with nl.scope as e:
        e.D[e.x, e.PROB[e.x]] = A[e.x] & B[e.x] & C[e.x]
        res = nl.query(tuple(), e.D[e.x, e.p])
    assert not res


def test_trivial_probability_query_result():
    nl = NeurolangPDL()
    b = set(((0.4, "a"), (0.5, "b")))
    B = nl.add_probabilistic_facts_from_tuples(b, name='B')
    with nl.scope as e:
        e.D[e.x, e.PROB[e.x]] = B[e.x]
        res = nl.query((e.p, e.x), e.D[e.x, e.p])
    assert set(res) == b


def test_equality():
    nl = NeurolangPDL()
    r1 = nl.add_tuple_set([(i,) for i in range(5)], name="r1")

    with nl.scope as e:
        e.r2[e.x] = r1(e.y) & (e.x == e.y * 2)

        sol = nl.query((e.x,), e.r2[e.x])

    assert set(sol) == set((2 * i,) for i in range(5))

    r2 = nl.add_tuple_set([("Hola",), ("Hello",), ("Bonjour",)], name="r2")

    @nl.add_symbol
    def lower(input: str) -> str:
        return input.lower()

    with nl.scope as e:
        e.r3[e.x] = r2(e.y) & (e.x == lower(e.y))

        sol = nl.query((e.x,), e.r3[e.x])

    assert set(sol) == set((("hola",), ("hello",), ("bonjour",)))


def test_equality2():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("Hola", "var"), ("Hello", "var2"), ("Bonjour", "var")],
        name="test_var",
    )

    @nl.add_symbol
    def word_lower(name: str) -> str:
        return str(name).lower()

    with nl.scope as e:
        e.low[e.lower] = e.test_var[e.name, "var"] & (
            e.lower == word_lower(e.name)
        )

        query = nl.query((e.lower,), e.low[e.lower])

    assert set(query) == set((("hola",), ("bonjour",)))


def test_result_both_deterministic_and_post_probabilistic():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            ("this", "is", "inglese"),
            ("questo", "è", "italiano"),
            ("ceci", "est", "francese"),
        ],
        name="parole",
    )
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.4, "valentino"),
            (0.6, "pietro"),
        ],
        name="persona_scelta",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.1, "italiano"),
            (0.7, "inglese"),
            (0.2, "francese"),
        ],
        name="lingua_parlata",
    )
    with nl.scope as e:
        e.e_una_lingua[e.lingua] = e.parole[e.subject, e.verb, e.lingua]
        e.lingua_da_lua_decisa[
            e.persona, e.lingua, e.PROB[e.persona, e.lingua]
        ] = (
            e.e_una_lingua[e.lingua]
            & e.persona_scelta[e.persona]
            & e.lingua_parlata[e.lingua]
        )
        e.utilizzare_le_probabilita[e.lingua, e.p] = e.lingua_da_lua_decisa[
            ..., e.lingua, e.p
        ] & (e.p > 0.1)
        res = nl.solve_all()
    assert "e_una_lingua" in res
    assert "lingua_da_lua_decisa" in res
    assert "utilizzare_le_probabilita" in res
    assert len(res["utilizzare_le_probabilita"]) == 3
    expected = RelationalAlgebraFrozenSet(
        [
            ("francese", 0.12),
            ("inglese", 0.7 * 0.4),
            ("inglese", 0.7 * 0.6),
        ]
    )
    assert_almost_equal(res["utilizzare_le_probabilita"], expected)


def test_result_query_relation_correct_column_names():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            ("alice",),
            ("bob",),
        ],
        name="person",
    )
    nl.add_tuple_set(
        [("alice", "paris"), ("bob", "marseille")],
        name="lives_in",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "bob"),
            (0.9, "alice"),
        ],
        name="does_not_smoke",
    )
    with nl.environment as e:
        e.climbs[e.p, e.PROB[e.p, e.city], e.city] = (
            e.person[e.p] & e.lives_in[e.p, e.city] & e.does_not_smoke[e.p]
        )
        solution = nl.solve_all()
    assert all(name in solution for name in ["climbs", "person", "lives_in"])


def test_add_constraints_and_rewrite():
    nl = NeurolangPDL()

    nl.add_tuple_set(
        [
            ("neurolang"),
        ],
        name="project",
    )

    nl.add_tuple_set(
        [
            ("neurolang", "db"),
            ("neurolang", "logic"),
        ],
        name="inArea",
    )

    with nl.scope as e:
        nl.add_constraint(
            e.project[e.x] & e.inArea[e.x, e.y],
            e.hasCollaborator[e.z, e.y, e.x],
        )

        e.p[e.b] = e.hasCollaborator[e.a, "db", e.b]
        res = nl.solve_all()

    assert list(res["p"])[0] == ("neurolang",)


def test_solve_marg_query():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            ("alice",),
            ("bob",),
        ],
        name="person",
    )
    nl.add_tuple_set(
        [("alice", "paris"), ("bob", "marseille")],
        name="lives_in",
    )
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.2, "alice", "running"),
            (0.8, "bob", "climbing"),
        ],
        name="practice",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "bob"),
            (0.9, "alice"),
        ],
        name="does_not_smoke",
    )
    with nl.environment as e:
        e.query[e.p, e.PROB[e.p, e.city, e.sport], e.city, e.sport] = (
            e.person[e.p] & e.lives_in[e.p, e.city] & e.does_not_smoke[e.p]
        ) // e.practice[e.p, e.sport]
        solution = nl.solve_all()
    assert all(name in solution for name in ["query", "person", "lives_in"])
    with nl.environment as e:
        result = nl.query(
            (e.p, e.prob, e.city, e.sport),
            e.query(e.p, e.prob, e.city, e.sport),
        )
    expected = RelationalAlgebraFrozenSet(
        iterable=[
            ("alice", 0.9, "paris", "running"),
            ("bob", 0.8, "marseille", "climbing"),
        ],
    )
    assert_almost_equal(result, expected)


def test_query_based_pfact():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            (2, 0.2),
            (7, 0.8),
            (4, 0.4),
        ],
        name="A",
    )
    with nl.environment as e:
        (e.B @ (e.p / 2))[e.x] = e.A[e.x, e.p]
        e.Query[e.PROB[e.x], e.x] = e.B[e.x]
        result = nl.query((e.x, e.p), e.Query[e.p, e.x])
    expected = RelationalAlgebraFrozenSet(
        [
            (2, 0.1),
            (7, 0.4),
            (4, 0.2),
        ]
    )
    assert_almost_equal(result, expected)


def test_query_based_pfact_empty():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            (2, 0.2),
            (7, 0.8),
            (4, 0.4),
        ],
        name="A",
    )
    with nl.environment as e:
        (e.B @ (e.p / 2))[e.x] = e.A[e.x, e.p] & (e.p > 0.8)
        e.Query[e.PROB[e.x], e.x] = e.B[e.x]
        result = nl.query((e.x, e.p), e.Query[e.p, e.x])
    expected = RelationalAlgebraFrozenSet(
        []
    )
    assert_almost_equal(result, expected)


def test_query_based_pfact_region_volume():
    nl = NeurolangPDL()

    @nl.add_symbol
    def volume(s: SphericalVolume) -> float:
        return (4 / 3) * np.pi * s.radius ** 3

    assert volume.symbol_name in nl.functions

    nl.add_tuple_set(
        [
            ("contained", SphericalVolume((0, 0, 0), 1)),
            ("container", SphericalVolume((0, 0, 0), 2)),
        ],
        name="my_sphere",
    )

    with nl.environment as e:
        (e.Z @ (e.volume[e.contained] / e.volume[e.container]))[
            e.contained, e.container
        ] = (
            e.my_sphere["contained", e.contained]
            & e.my_sphere["container", e.container]
        )
        e.Query[
            e.contained, e.container, e.PROB[e.contained, e.container]
        ] = e.Z[e.contained, e.container]
        res = nl.query((e.p,), e.Query[e.contained, e.container, e.p])
    expected = NamedRelationalAlgebraFrozenSet(("p",), [(1 / 2 ** 3,)])
    assert_almost_equal(res, expected)


def test_solve_marg_query_disjunction():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            ("alice",),
            ("bob",),
        ],
        name="person",
    )
    nl.add_tuple_set(
        [("alice", "paris"), ("bob", "marseille")],
        name="lives_in",
    )
    nl.add_probabilistic_choice_from_tuples(
        [
            (0.2, "alice", "running"),
            (0.8, "bob", "climbing"),
        ],
        name="practice",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.8, "bob"),
            (0.9, "alice"),
        ],
        name="does_not_smoke",
    )
    with pytest.raises(ForbiddenConditionalQueryNonConjunctive):
        with nl.environment as e:
            e.query[e.p, e.PROB[e.p, e.city, e.sport], e.city, e.sport] = (
                e.person[e.p]
                & (e.lives_in[e.p, e.city] | e.does_not_smoke[e.p])
            ) // e.practice[e.p, e.sport]


def test_query_without_safe_plan():
    nl = NeurolangPDL()
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "alice"),
            (0.8, "bob"),
        ],
        name="names",
    )

    with nl.environment as e:
        e.q[e.x, e.y, e.PROB[e.x, e.y]] = e.names[e.x] & e.names[e.y]

    with pytest.raises(dalvi_suciu_lift.NonLiftableException):
        res = nl.solve_all()


def test_query_without_safe_fails():
    nl = NeurolangPDL(
        probabilistic_solvers=(dalvi_suciu_lift.solve_succ_query,),
        probabilistic_marg_solvers=(dalvi_suciu_lift.solve_marg_query,),
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.2, "alice"),
            (0.8, "bob"),
        ],
        name="names",
    )

    with nl.environment as e:
        e.q[e.x, e.y, e.PROB[e.x, e.y]] = e.names[e.x] & e.names[e.y]

    with pytest.raises(dalvi_suciu_lift.NonLiftableException):
        nl.solve_all()


def test_cbma_two_term_conjunctive_query():
    nl = NeurolangPDL()
    nl.add_uniform_probabilistic_choice_over_set(
        [
            ("s1",),
            ("s2",),
            ("s3",),
        ],
        name="SelectedStudy",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.1, "t1", "s1"),
            (0.2, "t2", "s1"),
            (0.3, "t1", "s2"),
            (0.4, "t2", "s2"),
            (0.5, "t1", "s3"),
            (0.6, "t2", "s3"),
        ],
        name="TermInStudy",
    )
    nl.add_probabilistic_facts_from_tuples(
        [
            (0.6, "v1", "s1"),
            (0.5, "v2", "s1"),
            (0.4, "v1", "s2"),
            (0.3, "v2", "s2"),
            (0.2, "v1", "s3"),
            (0.1, "v2", "s3"),
        ],
        name="VoxelReported",
    )
    with nl.environment as e:
        e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
        e.Activation[e.v] = e.SelectedStudy[e.s] & e.VoxelReported[e.v, e.s]
        e.probmap[e.v, e.PROB[e.v]] = (e.Activation[e.v]) // (
            e.TermAssociation["t1"] & e.TermAssociation["t2"]
        )
        res = nl.query((e.v, e.p), e.probmap[e.v, e.p])
    expected = RelationalAlgebraFrozenSet(
        [
            (
                "v1",
                (0.6 * 0.1 * 0.2 + 0.4 * 0.3 * 0.4 + 0.2 * 0.5 * 0.6)
                / (0.1 * 0.2 + 0.3 * 0.4 + 0.5 * 0.6),
            ),
            (
                "v2",
                (0.5 * 0.1 * 0.2 + 0.3 * 0.4 * 0.3 + 0.1 * 0.5 * 0.6)
                / (0.1 * 0.2 + 0.3 * 0.4 + 0.5 * 0.6),
            ),
        ]
    )
    assert_almost_equal(res, expected)


def test_query_based_spatial_prior():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [
            (5, 5, 5, "s1"),
            (7, 5, 5, "s1"),
            (5, 5, 5, "s2"),
            (10, 10, 10, "s2"),
            (10, 10, 11, "s2"),
        ],
        name="FocusReported",
    )
    nl.add_tuple_set(
        list(itertools.product(range(15), range(15), range(15))), name="Voxel"
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)], name="SelectedStudy"
    )
    exp = nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])
    with nl.environment as e:
        (e.VoxelReported @ e.max(exp(-(e.d ** 2) / 5.0)))[
            e.i1, e.j1, e.k1, e.s
        ] = (
            e.FocusReported(e.i2, e.j2, e.k2, e.s)
            & e.Voxel(e.i1, e.j1, e.k1)
            & (e.d == e.EUCLIDEAN(e.i1, e.j1, e.k1, e.i2, e.j2, e.k2))
            & (e.d < 1)
        )
        e.Activation[e.i, e.j, e.k, e.PROB[e.i, e.j, e.k]] = e.VoxelReported(
            e.i, e.j, e.k, e.s
        ) & e.SelectedStudy(e.s)
        result = nl.query(
            (e.i, e.j, e.k, e.p), e.Activation(e.i, e.j, e.k, e.p)
        )
    expected = RelationalAlgebraFrozenSet(
        [
            (4, 5, 5, np.exp(-1 / 5)),
            (5, 4, 5, np.exp(-1 / 5)),
            (5, 5, 4, np.exp(-1 / 5)),
            (5, 5, 5, 1.0),
            (5, 5, 6, np.exp(-1 / 5)),
            (5, 6, 5, np.exp(-1 / 5)),
            (6, 5, 5, np.exp(-1 / 5)),
            (5, 6, 5, np.exp(-1 / 5)),
            (6, 5, 5, np.exp(-1 / 5)),
            (7, 4, 5, np.exp(-1 / 5) / 2),
            (7, 5, 4, np.exp(-1 / 5) / 2),
            (7, 5, 5, 1 / 2),
            (7, 5, 6, np.exp(-1 / 5) / 2),
            (7, 6, 5, np.exp(-1 / 5) / 2),
            (8, 5, 5, np.exp(-1 / 5) / 2),
            (9, 10, 10, np.exp(-1 / 5) / 2),
            (9, 10, 11, np.exp(-1 / 5) / 2),
            (10, 9, 10, np.exp(-1 / 5) / 2),
            (10, 9, 11, np.exp(-1 / 5) / 2),
            (10, 10, 9, np.exp(-1 / 5) / 2),
            (10, 10, 10, 0.5),
            (10, 10, 11, 0.5),
            (10, 10, 12, np.exp(-1 / 5) / 2),
            (10, 11, 10, np.exp(-1 / 5) / 2),
            (10, 11, 11, np.exp(-1 / 5) / 2),
            (11, 10, 10, np.exp(-1 / 5) / 2),
            (11, 10, 11, np.exp(-1 / 5) / 2),
        ]
    )
    assert_almost_equal(result, expected)


def test_simple_sigmoid():
    nl = NeurolangPDL()
    unnormalised = nl.add_tuple_set(
        [
            ("a", 1),
            ("b", 2),
            ("c", 7),
        ],
        name="unnormalised",
    )
    exp = nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])
    with nl.environment as e:
        (e.prob @ (1 / (1 + exp(-e.x))))[e.l] = unnormalised[e.l, e.x]
        e.Query[e.l, e.PROB[e.l]] = e.prob[e.l]
        res = nl.query((e.l, e.p), e.Query[e.l, e.p])
    expected = {(l, (1 / (1 + np.exp(-x)))) for l, x in unnormalised.value}
    assert_almost_equal(res, expected)


def test_postprob_conjunct_with_wlq_result():
    nl = NeurolangPDL()
    nl.add_tuple_set(
        [("s1",), ("s2",)],
        name="S",
    )
    nl.add_uniform_probabilistic_choice_over_set(
        [("s1",), ("s2",)],
        name="SS",
    )
    nl.add_tuple_set(
        [("t1", "s1"), ("t1", "s2"), ("t2", "s2")],
        name="TIS",
    )

    with nl.environment as e:
        e.TheWLQ[e.t, e.PROB(e.t)] = e.SS(e.s) & e.TIS(e.t, e.s)
        e.P[e.t, e.count(e.s)] = e.TIS(e.t, e.s)
        e.Q[e.count(e.s)] = e.S(e.s)
        e.TheQuery[e.t, e.N, e.m, e.p] = (
            e.TheWLQ(e.t, e.p) & e.P(e.t, e.m) & e.Q(e.N)
        )
        sol = nl.query((e.t, e.N, e.m, e.p), e.TheQuery(e.t, e.N, e.m, e.p))
    expected = {
        ("t1", 2, 2, 1.0),
        ("t2", 2, 1, 0.5),
    }
    assert_almost_equal(sol, expected)


def test_no_tuple_unicity_qbased_pfact():
    nl = NeurolangPDL(check_qbased_pfact_tuple_unicity=True)
    nl.add_tuple_set([(0.2, "a"), (0.5, "b"), (0.9, "a")], name="P")
    with nl.environment as e:
        (e.Q @ e.p)[e.x] = e.P(e.p, e.x)
        e.Query[e.x, e.PROB(e.x)] = e.Q(e.x)
        with pytest.raises(RepeatedTuplesInProbabilisticRelationError):
            nl.query((e.x, e.p), e.Query(e.x, e.p))
    nl = NeurolangPDL(check_qbased_pfact_tuple_unicity=False)
    nl.add_tuple_set([(0.2, "a"), (0.5, "b"), (0.9, "a")], name="P")
    with nl.environment as e:
        (e.Q @ e.p)[e.x] = e.P(e.p, e.x)
        e.Query[e.x, e.PROB(e.x)] = e.Q(e.x)
        result = nl.query((e.x, e.p), e.Query(e.x, e.p))
    expected = {("a", 0.2), ("b", 0.5), ("a", 0.9)}
    assert_almost_equal(result, expected)


def test_qbased_pfact_max_prob():
    nl = NeurolangPDL()
    nl.add_tuple_set([(0.2, "a"), (0.5, "b"), (0.9, "a")], name="P")
    with nl.environment as e:
        (e.Q @ e.max(e.p))[e.x] = e.P(e.p, e.x)
        e.Query[e.x, e.PROB(e.x)] = e.Q(e.x)
        sol = nl.query((e.x, e.p), e.Query(e.x, e.p))
    expected = {("a", 0.9), ("b", 0.5)}
    assert_almost_equal(sol, expected)
