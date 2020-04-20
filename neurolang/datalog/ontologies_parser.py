import nibabel as nib
import pandas as pd
import rdflib
from nilearn import datasets
from rdflib import OWL, RDF, RDFS, BNode

from ..exceptions import NeuroLangNotImplementedError
from ..expressions import Constant, ExpressionBlock, Symbol
from ..logic import Conjunction, LogicOperator, Union


class RightImplication(LogicOperator):
    """This class defines implications to the right. They are used to define
    constraints derived from ontologies. The functionality is the same as
    that of an implication, but with body and head inverted in position"""

    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
        self._symbols = consequent._symbols | antecedent._symbols

    def __repr__(self):
        return "RightImplication{{{} \u2192 {}}}".format(
            repr(self.antecedent), repr(self.consequent)
        )


class OntologyParser:
    def __init__(self, paths, load_format="xml"):
        self.namespaces_dic = None
        self.owl_dic = None
        if isinstance(paths, list):
            self._load_ontology(paths, load_format)
        else:
            self._load_ontology([paths], load_format)

        self._triple = Symbol.fresh()
        self._pointer = Symbol.fresh()
        self._dom = Symbol.fresh()

        self.parsed_restrictions = [
            OWL.allValuesFrom,
            OWL.hasValue,
            OWL.minCardinality,
            OWL.maxCardinality,
            OWL.cardinality,
        ]

    def _load_ontology(self, paths, load_format):
        self._create_graph(paths, load_format)

    def _create_graph(self, paths, load_format):
        g = rdflib.Graph()
        for path in paths:
            g.load(path, format=load_format)

        self.graph = g

    def parse_ontology(self, neurolangDL):
        self.eb = ExpressionBlock(())
        self.neurolangDL = neurolangDL

        self._load_domain()
        self._load_properties()
        self._load_constraints()

        self.neurolangDL.load_constraints(self.eb)
        return self.neurolangDL

    def _load_domain(self):
        pointers = map(
            lambda x: (str(x),),
            filter(lambda x: isinstance(x, BNode), set(self.graph.subjects())),
        )

        triples = map(
            lambda x: (str(x[0]), str(x[1]), str(x[2])), self.get_triples()
        )

        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        dom1 = RightImplication(self._triple(x, y, z), self._dom(x))
        dom2 = RightImplication(self._triple(x, y, z), self._dom(y))
        dom3 = RightImplication(self._triple(x, y, z), self._dom(z))

        self.neurolangDL.add_extensional_predicate_from_tuples(
            self._triple, triples
        )
        self.neurolangDL.add_extensional_predicate_from_tuples(
            self._pointer, pointers
        )

        self.eb = ExpressionBlock(self.eb.expressions + (dom1, dom2, dom3))

    def _load_properties(self):
        x = Symbol("x")
        z = Symbol("z")

        symbols = ()
        for pred in set(self.graph.predicates()):
            symbol_name = str(pred)
            symbol = Symbol(symbol_name)
            const = Constant(symbol_name)
            symbols += (
                RightImplication(self._triple(x, const, z), symbol(x, z)),
            )

        self.eb = ExpressionBlock(self.eb.expressions + symbols)

        self._parse_subproperties()
        self._parse_subclasses()
        self._parse_disjoint()

    def _parse_subproperties(self):
        rdf_schema_subPropertyOf = Symbol(str(RDFS.subPropertyOf))
        rdf_schema_subPropertyOf2 = Symbol(str(RDFS.subPropertyOf) + "2")

        w = Symbol("w")
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        subProperty = RightImplication(
            rdf_schema_subPropertyOf2(x, y), rdf_schema_subPropertyOf(x, y)
        )
        subProperty2 = RightImplication(
            Conjunction(
                (
                    rdf_schema_subPropertyOf2(x, y),
                    rdf_schema_subPropertyOf(y, z),
                )
            ),
            rdf_schema_subPropertyOf(x, z),
        )

        owl_inverseOf = Symbol(str(OWL.inverseOf))
        inverseOf = RightImplication(
            Conjunction(
                (
                    rdf_schema_subPropertyOf(x, y),
                    owl_inverseOf(w, x),
                    owl_inverseOf(z, y),
                )
            ),
            rdf_schema_subPropertyOf(w, z),
        )

        rdf_syntax_ns_type = Symbol(str(RDF.type))
        objectProperty = RightImplication(
            rdf_syntax_ns_type(x, Constant(str(OWL.ObjectProperty))),
            rdf_schema_subPropertyOf(x, x),
        )

        self.eb = ExpressionBlock(
            self.eb.expressions
            + (subProperty, subProperty2, inverseOf, objectProperty)
        )

    def _parse_subclasses(self):
        rdf_schema_subClassOf = Symbol(str(RDFS.subClassOf))
        rdf_schema_subClassOf2 = Symbol(str(RDFS.subClassOf) + "2")
        w = Symbol("w")
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        subClass = RightImplication(
            rdf_schema_subClassOf2(x, y), rdf_schema_subClassOf(x, y)
        )
        subClass2 = RightImplication(
            Conjunction(
                (rdf_schema_subClassOf2(x, y), rdf_schema_subClassOf(y, z))
            ),
            rdf_schema_subClassOf(x, z),
        )

        rdf_syntax_ns_rest = Symbol(str(RDF.rest))
        ns_rest = RightImplication(
            Conjunction(
                (
                    rdf_schema_subClassOf(x, y),
                    rdf_syntax_ns_rest(w, x),
                    rdf_syntax_ns_rest(z, y),
                )
            ),
            rdf_schema_subClassOf(w, z),
        )

        rdf_syntax_ns_type = Symbol(str(RDF.type))
        class_sim = RightImplication(
            rdf_syntax_ns_type(x, Constant(str(OWL.Class))),
            rdf_schema_subClassOf(x, x),
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (subClass, subClass2, ns_rest, class_sim)
        )

    def _parse_disjoint(self):
        w = Symbol("w")
        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        owl_disjointWith = Symbol(str(OWL.disjointWith))
        rdf_schema_subClassOf = Symbol(str(RDFS.subClassOf))
        disjoint = RightImplication(
            Conjunction(
                (
                    owl_disjointWith(x, y),
                    rdf_schema_subClassOf(w, x),
                    rdf_schema_subClassOf(z, y),
                )
            ),
            owl_disjointWith(w, z),
        )

        self.eb = ExpressionBlock(self.eb.expressions + (disjoint,))

    def _load_constraints(self):
        restriction_ids = []
        for s, _, _ in self.graph.triples((None, None, OWL.Restriction)):
            restriction_ids.append(s)

        for rest in restriction_ids:
            cut_graph = list(self.graph.triples((rest, None, None)))
            res_type = self._identify_restriction_type(cut_graph)

            try:
                process_restriction_method = getattr(
                    self, f"_process_{res_type}"
                )
                process_restriction_method(cut_graph)
            except AttributeError:
                raise NeuroLangNotImplementedError(
                    f"""Ontology parser doesn\'t handle
                    restrictions of type {res_type}"""
                )

    def _identify_restriction_type(self, list_of_triples):
        for triple in list_of_triples:
            if triple[1] == OWL.onProperty or triple[1] == RDF.type:
                continue
            else:
                return triple[1].rsplit("#")[-1]

        return ""

    def _process_hasValue(self, cut_graph):
        """
        A restriction containing a owl:hasValue constraint describes a class
        of all individuals for which the property concerned has at least
        one value semantically equal to V (it may have other values as well)

        The following example describes the class of individuals
        who have the individual referred to as Clinton as their parent:

        <owl:Restriction>
            <owl:onProperty rdf:resource="#hasParent" />
            <owl:hasValue rdf:resource="#Clinton" />
        </owl:Restriction>
        """
        parsed_prop, restricted_node, value = self._parse_restriction_nodes(
            cut_graph
        )

        rdf_type = Symbol(str(RDF.type))
        property_symbol = Symbol(parsed_prop)

        x = Symbol("x")

        constraint = ExpressionBlock(
            (
                RightImplication(
                    rdf_type(x, Constant(str(restricted_node))),
                    property_symbol(x, Constant(str(value))),
                ),
            )
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (constraint.expressions)
        )

    def _process_minCardinality(self, cut_graph):
        """
        A restriction containing an owl:minCardinality constraint describes
        a class of all individuals that have at least N semantically distinct
        values (individuals or data values) for the property concerned,
        where N is the value of the cardinality constraint.

        The following example describes a class of individuals
        that have at least two parents:

        <owl:Restriction>
            <owl:onProperty rdf:resource="#hasParent" />
            <owl:minCardinality rdf:datatype="&xsd;nonNegativeInteger">
                2
            </owl:minCardinality>
        </owl:Restriction>
        
        Note that an owl:minCardinality of one or more means that all
        instances of the class must have a value for the property.
        """

        parsed_prop, restricted_node, value = self._parse_restriction_nodes(
            cut_graph
        )

        rdf_type = Symbol(str(RDF.type))
        property_symbol = Symbol(parsed_prop)

        x = Symbol("x")
        y = Symbol("y")
        count = Symbol("count")

        constraint = ExpressionBlock(
            (
                RightImplication(
                    Union(
                        (
                            rdf_type(x, Constant(str(restricted_node))),
                            property_symbol(x, y),
                        )
                    ),
                    len(y) >= value,
                ),
            )
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (constraint.expressions)
        )

    def _process_maxCardinality(self, cut_graph):
        """
        A restriction containing an owl:maxCardinality constraint describes
        a class of all individuals that have at most N semantically distinct
        values (individuals or data values) for the property concerned,
        where N is the value of the cardinality constraint.

        The following example describes a class of individuals
        that have at most two parents:

        <owl:Restriction>
            <owl:onProperty rdf:resource="#hasParent" />
            <owl:maxCardinality rdf:datatype="&xsd;nonNegativeInteger">
                2
            </owl:maxCardinality>
        </owl:Restriction>
        """

        parsed_prop, restricted_node, value = self._parse_restriction_nodes(
            cut_graph
        )

        rdf_type = Symbol(str(RDF.type))
        property_symbol = Symbol(parsed_prop)

        x = Symbol("x")
        y = Symbol("y")

        constraint = ExpressionBlock(
            (
                RightImplication(
                    Union(
                        (
                            rdf_type(x, Constant(str(restricted_node))),
                            property_symbol(x, y),
                        )
                    ),
                    len(y) <= value,
                ),
            )
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (constraint.expressions)
        )

    def _process_cardinality(self, cut_graph):
        """
        A restriction containing an owl:cardinality constraint describes
        a class of all individuals that have exactly N semantically distinct
        values (individuals or data values) for the property concerned,
        where N is the value of the cardinality constraint.

        This construct is in fact redundant as it can always be replaced
        by a pair of matching owl:minCardinality and owl:maxCardinality
        constraints with the same value. It is included as a convenient
        shorthand for the user.

        The following example describes a class of individuals that have
        exactly two parents:

        <owl:Restriction>
            <owl:onProperty rdf:resource="#hasParent" />
            <owl:cardinality rdf:datatype="&xsd;nonNegativeInteger">
                2
            </owl:cardinality>
        </owl:Restriction>
        """
        parsed_prop, restricted_node, value = self._parse_restriction_nodes(
            cut_graph
        )

        rdf_type = Symbol(str(RDF.type))
        property_symbol = Symbol(parsed_prop)

        x = Symbol("x")
        y = Symbol("y")

        constraint = ExpressionBlock(
            (
                RightImplication(
                    Union(
                        (
                            rdf_type(x, Constant(str(restricted_node))),
                            property_symbol(x, y),
                        )
                    ),
                    len(y) == value,
                ),
            )
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (constraint.expressions)
        )

    def _process_allValuesFrom(self, cut_graph):
        """
        AllValuesFrom defines a class of individuals x
        for which holds that if the pair (x,y) is an instance of
        P (the property concerned), then y should be an instance
        of the class description.

        <owl:Restriction>
            <owl:onProperty rdf:resource="#hasParent" />
            <owl:allValuesFrom rdf:resource="#Human"  />
        </owl:Restriction>
        
        This example describes an anonymous OWL class of all individuals
        for which the hasParent property only has values of class Human
        """

        parsed_prop, restricted_node, values = self._parse_restriction_nodes(
            cut_graph
        )

        allValuesFrom = self._parse_list(values)

        constraints = ExpressionBlock(())

        property_symbol = Symbol(parsed_prop)
        rdf_type = Symbol(str(RDF.type))
        x = Symbol("x")
        y = Symbol("y")

        for value in allValuesFrom:
            constraints = ExpressionBlock(
                constraints.expressions
                + (
                    RightImplication(
                        Conjunction(
                            (
                                rdf_type(y, Constant(str(restricted_node))),
                                property_symbol(y, x),
                            )
                        ),
                        rdf_type(x, Constant(str(value))),
                    ),
                )
            )

        self.eb = ExpressionBlock(
            self.eb.expressions + constraints.expressions
        )

    def _parse_restriction_nodes(self, cut_graph):

        restricted_node = cut_graph[0][0]
        restricted_node = list(
            self.graph.triples((None, None, restricted_node))
        )[0][0]
        for triple in cut_graph:
            if OWL.onProperty == triple[1]:
                parsed_property = str(triple[2])
            elif triple[1] in self.parsed_restrictions:
                value = triple[2]

        return parsed_property, restricted_node, value

    def _parse_list(self, initial_node):
        list_node = RDF.nil
        values = []
        for node_triples in self.graph.triples((initial_node, None, None)):
            if OWL.unionOf == node_triples[1]:
                list_node = node_triples[2]
            else:
                values.append(node_triples[0])

        while list_node != RDF.nil:
            list_iter = self.graph.triples((list_node, None, None))
            values.append(self._get_list_first_value(list_iter))
            list_node = self._get_list_rest_value(list_iter)

        return values

    def _get_list_first_value(self, list_iter):
        for triple in list_iter:
            if RDF.first == triple[1]:
                return triple[2]

    def _get_list_rest_value(self, list_iter):
        for triple in list_iter:
            if RDF.rest == triple[1]:
                return triple[2]

    def get_triples(self):
        return self.graph.triples((None, None, None))
