import io
from typing import AbstractSet, Tuple

import numpy as np
import pytest

from ...exceptions import UnsupportedProgramError, UnsupportedQueryError
from ...probabilistic.exceptions import (
    ForbiddenConditionalQueryNonConjunctive,
    UnsupportedProbabilisticQueryError
)
from ...utils.relational_algebra_set import RelationalAlgebraFrozenSet
from ..probabilistic_frontend import ProbabilisticFrontend


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
    nl = ProbabilisticFrontend()

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


def test_deterministic_query():
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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

    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
    P = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("b",), ("c",)], name="P"
    )
    Q = nl.add_uniform_probabilistic_choice_over_set(
        [("a",), ("d",), ("c",)], name="Q"
    )
    with nl.scope as e:
        e.Z[e.x, e.PROB[e.x]] = P[e.x] & Q[e.x]
        res = nl.query((e.x, e.p), e.Z[e.x, e.p])
    expected = RelationalAlgebraFrozenSet(
        [
            ("a", 1 / 9),
            ("c", 1 / 9),
        ]
    )
    assert_almost_equal(res, expected)


def test_solve_query_prob_col_not_last():
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    neurolang = ProbabilisticFrontend()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")

    dataset = {(i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset, name="q")
    s[x, y] = ~q(x, x) & q(x, y)

    res = neurolang.solve_all()

    assert res["s"].to_unnamed() == {(i, j) for i, j in dataset if i != j}


@pytest.mark.xfail(reason="RAP negation needs to be implemented")
def test_neurolange_dl_probabilistic_negation():
    neurolang = ProbabilisticFrontend()
    s = neurolang.new_symbol(name="s")
    x = neurolang.new_symbol(name="x")
    y = neurolang.new_symbol(name="y")
    prob = neurolang.new_symbol(name="PROB")

    dataset_det = {(i, i * 2) for i in range(10)}
    dataset = {((1 + i) / 10, i, i * 2) for i in range(10)}
    q = neurolang.add_tuple_set(dataset_det, name="q")
    r = neurolang.add_probabilistic_facts_from_tuples(dataset, name="r")

    s[x, y, prob(x, y)] = ~r(x, x) & q(x, y)

    res = neurolang.solve_all()

    assert res["s"].to_unnamed() == {(i, j) for i, j in dataset if i != j}


def test_neurolang_dl_aggregation():
    neurolang = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    assert res == {("a", 0.2 * 0.2 + 0.2 * 0.1), ("b", 0.9 * 0.7)}


def test_empty_result_query():
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
    A = nl.add_tuple_set([("f",), ("d",)], name="A")
    B = nl.add_probabilistic_facts_from_tuples(
        [(0.2, "a"), (0.7, "b"), (0.6, "c")], name="B"
    )
    with pytest.raises(UnsupportedQueryError):
        with nl.scope as e:
            e.Q[e.x] = A[e.x] & B[e.x] & A["b"]
            nl.query((e.x,), e.Q[e.x])


def test_empty_boolean_query_result():
    nl = ProbabilisticFrontend()
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


def test_equality():
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()
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
    nl = ProbabilisticFrontend()

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
    nl = ProbabilisticFrontend()
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


def test_solve_marg_query_disjunction():
    nl = ProbabilisticFrontend()
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
                e.person[e.p] & (
                    e.lives_in[e.p, e.city] |
                    e.does_not_smoke[e.p]
                )
            ) // e.practice[e.p, e.sport]
