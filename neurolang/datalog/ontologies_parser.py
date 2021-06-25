import warnings

import rdflib
from rdflib import BNode, Literal
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from ..exceptions import NeuroLangException, NeuroLangNotImplementedError
from ..expressions import Constant, Symbol
from ..logic import Conjunction, Implication
from .constraints_representation import RightImplication


class OntologyParser:
    """
    This class is in charge of generating the rules that can be derived
    from an ontology, both at entity and constraint levels.
    """

    STRUCTURAL_KNOWLEDGE_NAMESPACE = 'neurolang:'

    def __init__(self, paths, load_format="xml"):
        if isinstance(paths, list):
            self._load_ontology(paths, load_format)
        else:
            self._load_ontology([paths], [load_format])

        self.parsed_constraints = {}
        self.estructural_knowledge = {}
        self.parsed_rules = {}
        self.existential_rules = {}

    def _load_ontology(self, paths, load_format):
        rdfGraph = rdflib.Graph()
        for counter, path in enumerate(paths):
            rdfGraph.load(path, format=load_format[counter])

        self.rdfGraph = rdfGraph

    def parse_ontology(self):
        '''
        Main function to be called for ontology processing.

        Returns
        -------
        Two dictionaries. One with contraints, for example, rules derived
        from restrictions of type someValuesFrom. The other one, with the
        implications derived from the rest of the ontological information,
        for example, the label property. Both dictionaries index lists of
        rules using the rule consequent functor.
        '''
        self._parse_classes()
        self._parse_related_individuals()

        return self.parsed_constraints, self.parsed_rules, self.estructural_knowledge

    def _parse_classes(self):
        '''This method obtains all the classes present in the ontology and
        iterates over them, parsing them according to the properties
        that compose them.
        '''
        _all_classes = self._get_all_classes()

        for class_name in list(_all_classes):
            for entity, prop, value in self.rdfGraph.triples((class_name, None, None)):

                if prop == RDF.type and value == OWL.Class:
                    # The triple used to index the class must be skipped.
                    continue

                elif prop == RDFS.subClassOf:
                    self._parseSubClass(entity, value)

                elif prop == OWL.equivalentClass:
                    self._parseEquivalentClass(entity, prop, value)

                elif prop == OWL.EnumeratedClass:
                    self._parseEnumeratedClass(entity, prop, value)

                elif prop == OWL.DisjointClass:
                    self._parseDisjointClass(entity, prop, value)
                else:
                    if not isinstance(value, BNode):
                        self._parse_property(entity, prop, value)
                    else:
                        raise NeuroLangNotImplementedError

    def _get_all_classes(self):
        '''
        Function in charge of obtaining all the classes of the ontology
        to iterate through at the time of parsing.

        Returns
        -------
        A set of URIs representing the classes that compose the ontology.
        '''
        classes = set()
        for s, _, _ in self.rdfGraph.triples((None, RDF.type, OWL.Class)):
            if not isinstance(s, BNode):
                classes.add(s)
        for s, _, _ in self.rdfGraph.triples((None, RDF.type, RDFS.Class)):
            if not isinstance(s, BNode):
                classes.add(s)

        return classes

    def _get_all_named_individual(self):
        individuals = set()
        for s, _, _ in self.rdfGraph.triples((None, RDF.type, OWL.NamedIndividual)):
            individuals.add(s)

        return individuals

    def _parse_related_individuals(self):
        _all_individuals = self._get_all_named_individual()

        for individual_name in list(_all_individuals):
            for entity, prop, value in self.rdfGraph.triples((individual_name, None, None)):

                if prop == RDF.type and value == OWL.NamedIndividual:
                    # The triple used to index the individual must be skipped.
                    continue

                if prop == SKOS.related:
                    self._parse_individual_related(entity, prop, value)



    def _parseSubClass(self, entity, val):
        '''Function in charge of identifying the type of constraint to be
        parsed. At the moment it allows to parse subclasses, basic
        constraints and a single level of intersections.
        Nested intersections are not yet implemented.

        Parameters
        ----------
        entity : URIRef
            entity to be parsed.

        value : URIRef or BNode
            value associated with the entity.
        '''
        res = []
        if isinstance(val, BNode):
            bnode_triples = list(self.rdfGraph.triples((val, None, None)))
            restriction_dic = {b: c for _, b, c in bnode_triples}
            if OWL.Restriction in restriction_dic.values():
                res = self._parse_restriction(entity, restriction_dic)
                self._categorize_constraints(res)
            elif OWL.intersectionOf in restriction_dic.keys():
                c = restriction_dic[OWL.intersectionOf]
                int_triples = list(self.rdfGraph.triples((c, None, None)))
                inter_entity = [
                    a[2] for a in int_triples if not isinstance(a[2], BNode)
                ]
                inter_BNode = [a[2] for a in int_triples if isinstance(a[2], BNode)]
                if len(inter_entity) == 1 and len(inter_BNode) == 1:
                    self._parse_BNode_intersection(
                        entity, inter_BNode[0], inter_entity[0]
                    )
                else:
                    warnings.warn("Complex intersectionOf are not implemented yet")

            elif OWL.unionOf in restriction_dic.keys():
                warnings.warn("Not implemented yet: unionOf")
            elif OWL.complementOf in restriction_dic.keys():
                warnings.warn("Not implemented yet: complementOf")
            else:
                raise NotImplementedError(f"Something went wrong: {restriction_dic}")
        else:
            ant = Symbol(self._parse_name(entity))
            cons = Symbol(self._parse_name(val))
            x = Symbol.fresh()
            #imp = RightImplication(ant(x), cons(x))
            imp = Implication(cons(x), ant(x))
            self._categorize_constraints([imp])

            neurolang_subclassof = Symbol(self.STRUCTURAL_KNOWLEDGE_NAMESPACE+'subClassOf')
            est = neurolang_subclassof(Constant(ant.name), Constant(cons.name))
            self._categorize_structural_knowledge([est])

    def _parse_BNode_intersection(self, entity, node, inter_entity):
        '''When the rules that compose a constraint are defined within an
        intersection, it needs to be manipulated in a special way.
        This is the method in charge of that behavior.

        Parameters
        ----------
        entity : URIRef
            the main entity containing the intersection.

        node : URIRef
            URI defining the intersection.

        inter_entity : URIRef
            the main entity defined within the intersection.

        '''
        triple_restriction = list(self.rdfGraph.triples((node, None, None)))
        nil = [a[2] for a in triple_restriction if not isinstance(a[2], BNode)]
        bnode = [a[2] for a in triple_restriction if isinstance(a[2], BNode)]
        support_prop = Symbol.fresh()
        if nil[0] == RDF.nil:
            bnode_triples = list(self.rdfGraph.triples((bnode[0], None, None)))
            restriction_dic = {b: c for _, b, c in bnode_triples}
            if OWL.Restriction in restriction_dic.values():
                res = self._parse_restriction(entity, restriction_dic, support_prop)
                self._categorize_constraints(res)
            con = Symbol(self._parse_name(inter_entity))
            x = Symbol.fresh()
            y = Symbol.fresh()
            imp = RightImplication(support_prop(x, y), con(x))
            self._categorize_constraints([imp])
        else:
            warnings.warn("Complex intersectionOf are not implemented yet")

    def _parse_name(self, obj):
        '''Function that transforms the names of the entities of the ontology
        while preserving the namespace to avoid conflicts.

        Example: the URI `http://www.w3.org/2004/02/skos/core#altLabel`
        becomes `core:altLabel`.


        Parameters
        ----------
        obj : URIRef
            entity to be renamed.

        Returns
        -------
        String with the new name associated to the entity.
        '''
        if isinstance(obj, Literal):
            return str(obj)

        obj_split = obj.split("#")
        if len(obj_split) == 2:
            name = obj_split[1]
            namespace = obj_split[0].split("/")[-1]
            if name[0] != "" and namespace != "":
                res = namespace + ":" + name
            else:
                res = name
        else:
            obj_split = obj.split("/")
            res = obj_split[-1]

        return res

    def _parse_restriction(self, entity, restriction_dic, support_prop=None):
        '''Method for the identification of the type of restriction.
        Each of the possible available restrictions has its own
        method in charge of information parsing.

        Parameters
        ----------
        entity : URIRef
            entity to be parsed.

        restriction_dic : dict
            dictionary containing the information of the constraint
            to be processed.

        support_prop : Symbol, default None
            Optional symbol. It is used to predefine the symbol used in
            rules with existentials, when necessary
            (mainly in nested definitions).

        Returns
        -------
        A list of rules that compose the parsed constraint.

        '''
        cons = []
        prop = restriction_dic[OWL.onProperty]

        if OWL.someValuesFrom in restriction_dic.keys():
            node = restriction_dic[OWL.someValuesFrom]
            if isinstance(node, BNode):
                node = self._solve_BNode(node)
            else:
                node = [node]
            cons = self._parse_someValuesFrom(
                entity, prop, node, support_prop=support_prop
            )

        elif OWL.allValuesFrom in restriction_dic.keys():
            node = restriction_dic[OWL.allValuesFrom]
            if isinstance(node, BNode):
                node = self._solve_BNode(node)
            else:
                node = [node]
            cons = self._parse_allValuesFrom(entity, prop, node)

        elif OWL.hasValue in restriction_dic.keys():
            node = restriction_dic[OWL.hasValue]
            if isinstance(node, BNode):
                node = self._solve_BNode(node)
            else:
                node = [node]
            cons = self._parse_hasValue(entity, prop, node)

        elif OWL.minCardinality in restriction_dic.keys():
            warnings.warn(
                "minCardinality constraints cannot be implemented in Datalog syntax"
            )
        elif OWL.maxCardinality in restriction_dic.keys():
            warnings.warn(
                "maxCardinality constraints cannot be implemented in Datalog syntax"
            )
        else:
            raise NeuroLangException(
                "This restriction does not correspond to an OWL DL constraint:",
                restriction_dic.keys(),
            )

        return cons

    def _parse_someValuesFrom(self, entity, prop, nodes, support_prop=None):
        '''After the restriction is identified as of type `someValuesFrom`,
        this function is in charge of parsing the information and returning it
        in the form of rules that our Datalog program can interpret.

        Parameters
        ----------
        entity : URIRef
            entity to be parsed.

        prop : URIRef
            propertie associated with the entity.

        nodes : list
            list of the values associated with the entity and the property.

        support_prop : Symbol, default None
            Optional symbol. It's used to predefine the symbol used in
            rules with existentials, when necessary
            (mainly in nested definitions).
        '''
        constraints = []
        if support_prop is None:
            support_prop = Symbol.fresh()
        x = Symbol.fresh()
        y = Symbol.fresh()
        onProp = Symbol(self._parse_name(prop))
        entity = Symbol(self._parse_name(entity))
        for value in nodes:
            value = self._parse_name(value)
            ext_rule = RightImplication(entity(x), support_prop(x, y))
            constraints.append(ext_rule)
            self._add_existential_rule(entity(x), ext_rule)
        prop_imp = RightImplication(support_prop(x, y), onProp(x, y))
        exists_imp = RightImplication(support_prop(x, y), Symbol(value)(y))
        # prop_imp = Implication(onProp(x, y), support_prop(x, y))
        # exists_imp = Implication(Symbol(value)(y), support_prop(x, y))

        constraints.append(exists_imp)
        constraints.append(prop_imp)

        return constraints

    def _parse_allValuesFrom(self, entity, prop, nodes):
        '''After the restriction is identified as of type `allValuesFrom`,
        this function is in charge of parsing the information and returning it
        in the form of rules that our Datalog program can interpret.

        Parameters
        ----------
        entity : URIRef
            entity to be parsed.

        prop : URIRef
            propertie associated with the entity.

        nodes : list
            list of the values associated with the entity and the property.
        '''
        constraints = []
        x = Symbol.fresh()
        y = Symbol.fresh()
        ant = Symbol(self._parse_name(entity))
        onProp = Symbol(self._parse_name(prop))
        conj = Conjunction((ant(x), onProp(x, y)))
        for value in nodes:
            value = self._parse_name(value)
            constraints.append(Implication(Symbol(value)(y), conj))
            # constraints.append(RightImplication(conj, Symbol(value)(y)))

        return constraints

    def _parse_hasValue(self, entity, prop, nodes):
        '''After the constraint is identified as of type `hasValue`,
        this function is in charge of parsing the information and returning it
        in the form of rules that our Datalog program can interpret.

        Parameters
        ----------
        entity : URIRef
            entity to be parsed.

        prop : URIRef
            propertie associated with the entity.

        nodes : list
            list of the values associated with the entity and the property.
        '''
        x = Symbol.fresh()
        ent = Symbol(self._parse_name(entity))
        onProp = Symbol(self._parse_name(prop))
        value = self._parse_name(nodes[0])
        return [Implication(Symbol(onProp)(x, Constant(value)), Symbol(ent)(x))]
        #return [RightImplication(Symbol(ent)(x), Symbol(onProp)(x, Constant(value)))]

    def _solve_BNode(self, initial_node):
        '''Once a BNode is identified, this function iterates over each of the pointers
         that compose it and returns a list with those values.

        Parameters
        ----------
        initial_node : BNode
            pointer to the first item in the list

        Returns
        -------
        A list of all the values that arise from traversing the list of BNodes.
         '''
        list_node = RDF.nil
        values = []
        for node_triples in self.rdfGraph.triples((initial_node, None, None)):
            if OWL.unionOf == node_triples[1]:
                list_node = node_triples[2]

        while list_node != RDF.nil and list_node is not None:
            list_iter = self.rdfGraph.triples((list_node, None, None))
            list_iter = list(list_iter)
            values.append(self._get_list_first_value(list_iter))
            list_node = self._get_list_rest_value(list_iter)

        return values

    def _get_list_first_value(self, list_iter):
        '''Used to iterate between pointers in triples, this function takes
        care of finding the pointed value and returning it.

        Parameters
        ----------
        list_iter : list
            list of triples

        Returns
        -------
            see description.
        '''
        for triple in list_iter:
            if RDF.first == triple[1]:
                return triple[2]

    def _get_list_rest_value(self, list_iter):
        '''Used to iterate between pointers in triples, this function takes
        care of finding the pointer to the next element and returning it.

        Parameters
        ----------
        list_iter : list
            list of triples

        Returns
        -------
            see description.
        '''
        for triple in list_iter:
            if RDF.rest == triple[1]:
                return triple[2]

    def _parse_property(self, entity, prop, value):
        '''Any rule that is not a restriction or a derivation on
        classes(as the case of subClassOf or EnumeratedClass) is a
        property that defines a field inside the entity with its
        corresponding information. For example, the properties
        label, altLabel, definition, etc.

        This function is in charge of parsing that information.

        Parameters
        ----------
        entity : URIRef
            entity to be parsed.

        prop : URIRef
            propertie associated with the entity.

        nodes : URIRef
            value associated with the entity and the property.
        '''
        entity = Symbol(self._parse_name(entity))
        entity_name = Constant(self._parse_name(value))
        x = Symbol.fresh()
        ant = entity(x)
        label = Symbol(self._parse_name(prop))
        con = label(x, entity_name)

        #self._categorize_constraints([RightImplication(ant, con)])
        self._categorize_constraints([Implication(con, ant)])

        prop_name = label.name.split(':')[-1]
        neurolang_prop = Symbol(self.STRUCTURAL_KNOWLEDGE_NAMESPACE+prop_name)
        est = neurolang_prop(Constant(entity.name), entity_name)
        self._categorize_structural_knowledge([est])

    def _parse_individual_related(self, entity, prop, value):
        entity = Symbol(self._parse_name(entity))
        entity_name = Constant(self._parse_name(value))
        label = Symbol(self._parse_name(prop))

        prop_name = label.name.split(':')[-1]
        neurolang_prop = Symbol(self.STRUCTURAL_KNOWLEDGE_NAMESPACE+prop_name)
        est = neurolang_prop(Constant(entity.name), entity_name)
        self._categorize_structural_knowledge([est])


    def _parseEnumeratedClass(self, entity, prop, value):
        warnings.warn("Not implemented yet: EnumeratedClass")

    def _parseEquivalentClass(self, entity, prop, value):
        # <->
        warnings.warn("Not implemented yet: EquivalentClass")

    def _parseDisjointClass(self, entity, prop, value):
        # inconsistent(C, D) :− disjointWith(C, D), type(X, C), type(X, D).
        warnings.warn("Not implemented yet: DisjointClass")

    def _categorize_constraints(self, formulas):
        for sigma in formulas:
            if isinstance(sigma, RightImplication):
                sigma_functor = sigma.consequent.functor.name
                if sigma_functor in self.parsed_constraints:
                    cons_set = self.parsed_constraints[sigma_functor]
                    cons_set.add(sigma)
                    self.parsed_constraints[sigma_functor] = cons_set
                else:
                    self.parsed_constraints[sigma_functor] = set([sigma])

            else:
                sigma_functor = Symbol(sigma.consequent.functor.name)
                if sigma_functor in self.parsed_rules:
                    cons_set = self.parsed_rules[sigma_functor]
                    cons_set.add(sigma)
                    self.parsed_rules[sigma_functor] = cons_set
                else:
                    self.parsed_rules[sigma_functor] = set([sigma])

    def _categorize_structural_knowledge(self, formulas):
        for sigma in formulas:
            sigma_functor = sigma.functor
            if sigma_functor in self.estructural_knowledge:
                cons_set = self.estructural_knowledge[sigma_functor]
                cons_set.add(sigma)
                self.estructural_knowledge[sigma_functor] = cons_set
            else:
                self.estructural_knowledge[sigma_functor] = set([sigma])


    def _add_existential_rule(self, entity, rule):
        if entity not in self.existential_rules:
            self.existential_rules[entity] = [rule]
        else:
            rules = self.existential_rules[entity]
            rules.append(rule)
            self.existential_rules[entity] = rules
