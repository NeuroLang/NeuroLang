import io

from ..probabilistic_frontend import ProbabilisticFrontend


def test_deterministic_query():

    pass


def test_probabilistic_query():

    pass


def test_mixed_queries():

    pass


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

    resp = list(solution_instance["answer"].value.unwrapped_iter())
    assert (
        "http://www.w3.org/2002/03owlt/hasValue/premises001#i",
        "true",
    ) in resp
