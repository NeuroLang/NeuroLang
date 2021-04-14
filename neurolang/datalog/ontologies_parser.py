import warnings

import rdflib
from rdflib import BNode, Literal
from rdflib.namespace import OWL,RDF, RDFS, SKOS

from ..exceptions import NeuroLangException, NeuroLangNotImplementedError
from ..expressions import Constant, Symbol
from ..logic import Conjunction
from .constraints_representation import RightImplication

class OntologyParser:
    """
    This class is in charge of generating the rules that can be derived
    from an ontology, both at entity and constraint levels.
    """


    def __init__(self, paths, load_format="xml"):
            #self.namespaces_dic = None
            #self.owl_dic = None
            if isinstance(paths, list):
                self._load_ontology(paths, load_format)
            else:
                self._load_ontology([paths], [load_format])

            self.parsed_constraints = []

    def _load_ontology(self, paths, load_format):
        rdfGraph = rdflib.Graph()
        for counter, path in enumerate(paths):
            rdfGraph.load(path, format=load_format[counter])

        self.rdfGraph = rdfGraph

    def parse_ontology(self):
        self._parse_classes()

        return self.parsed_constraints

    def _parse_classes(self):
        _all_classes = self._get_all_classes()

        for class_name in list(_all_classes):
            for entity, prop, value in self.rdfGraph.triples((class_name, None, None)):
                
                if prop == RDF.type and value == OWL.Class:
                    # The triple used to index the class must be skipped.
                    continue
                
                elif prop == RDFS.subClassOf:
                    self._parseSubClass(entity, value)
                
                elif prop == RDFS.label:
                    self._parseLabel(entity, prop, value)
                    
                elif prop == OWL.equivalentClass:
                    self._parseEquivalentClass(entity, prop,  value)
                    
                elif prop == OWL.EnumeratedClass:
                    self._parseEnumeratedClass(entity, prop, value)

                elif prop == OWL.DisjointClass:
                    self._parseDisjointClass(entity, prop, value)
                
                elif prop == SKOS.prefLabel:
                    self._parsePrefLabel(entity, prop, value)

                elif prop == SKOS.hasTopConcept:
                    self._parseHasTopConcept(entity, prop, value)

                elif prop == SKOS.altLabel:
                    self._parseAltLabel(entity, prop, value) 
                else:
                    if not isinstance(value, BNode):
                        self._parseProperty(entity, prop, value)
                    else:
                        raise NeuroLangNotImplementedError


    def _get_all_classes(self):
        classes = set()
        for s, _, _ in self.rdfGraph.triples((None, RDF.type , OWL.Class)):
            if not isinstance(s, BNode):
                classes.add(s)
        for s, _, _ in self.rdfGraph.triples((None, RDF.type , RDFS.Class)):
            if not isinstance(s, BNode):
                classes.add(s)

        return classes

    def _parseSubClass(self, entity, val):
        res = []
        if isinstance(val, BNode):
            bnode_triples = list(self.rdfGraph.triples((val, None, None)))
            restriction_dic = {b:c for _, b, c in bnode_triples}
            if OWL.Restriction in restriction_dic.values():
                res = self._parse_restriction(entity, restriction_dic)
                self.parsed_constraints = self.parsed_constraints + res
            elif OWL.intersectionOf in restriction_dic.keys():
                c = restriction_dic[OWL.intersectionOf]
                int_triples = list(self.rdfGraph.triples((c, None, None)))
                inter_entity = [a[2] for a in int_triples if not isinstance(a[2], BNode)]
                inter_BNode = [a[2] for a in int_triples if isinstance(a[2], BNode)]
                if len(inter_entity) == 1 and len(inter_BNode) == 1:
                        self._parse_BNode_intersection(entity, inter_BNode[0], inter_entity[0])
                else:
                    warnings.warn('Complex intersectionOf are not implemented yet')
                
            elif OWL.unionOf in restriction_dic.keys():
                warnings.warn('Not implemented yet: unionOf')
            elif OWL.complementOf in restriction_dic.keys():
                warnings.warn('Not implemented yet: complementOf')
            else:
                raise NotImplementedError(f'Something went wrong: {restriction_dic}')
        else:
            cons = Symbol(self._parse_name(entity))
            ant = Symbol(self._parse_name(val))
            x = Symbol.fresh()
            imp = RightImplication(ant(x), cons(x))
            self.parsed_constraints.append(imp)

        return res


    def _parse_BNode_intersection(self, entity, node, inter_entity):
        triple_restriction = list(self.rdfGraph.triples((node, None, None)))
        nil = [a[2] for a in triple_restriction if not isinstance(a[2], BNode)]
        bnode = [a[2] for a in triple_restriction if isinstance(a[2], BNode)]
        support_prop = Symbol.fresh()
        if nil[0] == RDF.nil:
            bnode_triples = list(self.rdfGraph.triples((bnode[0], None, None)))
            restriction_dic = {b:c for _, b, c in bnode_triples}
            if OWL.Restriction in restriction_dic.values():
                res = self._parse_restriction(entity, restriction_dic, support_prop)
                self.parsed_constraints = self.parsed_constraints + res

            con = Symbol(self._parse_name(inter_entity))
            x = Symbol.fresh()
            y = Symbol.fresh()
            imp = RightImplication(support_prop(x, y), con(x))
            self.parsed_constraints.append(imp)
        else:
            warnings.warn('Complex intersectionOf are not implemented yet')

    def _parse_name(self, obj):
        if isinstance(obj, Literal):
            return str(obj)

        obj_split = obj.split('#')
        if len(obj_split) == 2:
            name = obj_split[1]
            namespace = obj_split[0].split('/')[-1]
            if name[0] != '' and namespace != '':
                res = namespace + ':' + name
            else:
                res = name
        else:
            obj_split = obj.split('/')
            res = obj_split[-1]

        return res 
                
    def _parse_restriction(self, entity, restriction_dic, support_prop=None):
        cons = []
        prop = restriction_dic[OWL.onProperty]
        
        if OWL.someValuesFrom in restriction_dic.keys():
            node = restriction_dic[OWL.someValuesFrom]
            if isinstance(node, BNode):
                node = self._solve_BNode(node)
            else:
                node = [node]
            cons = self._parse_someValuesFrom(entity, prop, node, support_prop=support_prop)
        
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
            warnings.warn('minCardinality constraints cannot be implemented in Datalog syntax')
        elif OWL.maxCardinality in restriction_dic.keys():
            warnings.warn('maxCardinality constraints cannot be implemented in Datalog syntax')
        else:
            raise NeuroLangException('This restriction does not correspond to an OWL DL constraint:', restriction_dic.keys())

        return cons

    def _parse_someValuesFrom(self, entity, prop, nodes, support_prop=None):
        constraints = []
        if support_prop is None:
            support_prop = Symbol.fresh()
        x = Symbol.fresh()
        y = Symbol.fresh()
        onProp = Symbol(self._parse_name(prop))
        entity = Symbol(self._parse_name(entity))
        for value in nodes:
            value = self._parse_name(value)
            constraints.append(RightImplication(support_prop(x, y), Symbol(value)(y)))
        prop_imp = RightImplication(support_prop(x, y), onProp(x, y))
        exists_imp = RightImplication(entity(x), support_prop(x, y))
        
        constraints.append(exists_imp)
        constraints.append(prop_imp)

        return constraints

    def _parse_allValuesFrom(self, entity, prop, nodes):
        constraints = []
        x = Symbol.fresh()
        y = Symbol.fresh()
        ant = Symbol(self._parse_name(entity))
        onProp = Symbol(self._parse_name(prop))
        conj = Conjunction((ant(x), onProp(x, y)))
        for value in nodes:
            value = self._parse_name(value)
            constraints.append(RightImplication(conj, Symbol(value)(y)))

        return constraints
        
    def _parse_hasValue(self, entity, prop, nodes):
        warnings.warn('Not implemented yet: complementOf')
        return []
            
    def _solve_BNode(self, initial_node):
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
        for triple in list_iter:
            if RDF.first == triple[1]:
                return triple[2]
            
    def _get_list_rest_value(self, list_iter):
        for triple in list_iter:
            if RDF.rest == triple[1]:
                return triple[2]
            
    def _parseLabel(self, entity, prop, value):
        entity = Symbol(self._parse_name(entity))
        entity_name = Constant(self._parse_name(value))
        x = Symbol.fresh()
        ant = entity(x)
        label = Symbol(self._parse_name(prop))
        con = label(x, entity_name)

        self.parsed_constraints.append(RightImplication(ant, con))
        
    def _parsePrefLabel(self, entity, prop, value):
        entity = Symbol(self._parse_name(entity))
        entity_name = Constant(self._parse_name(value))
        x = Symbol.fresh()
        ant = entity(x)
        label = Symbol(self._parse_name(prop))
        con = label(x, entity_name)
        self.parsed_constraints.append(RightImplication(ant, con))
        
    def _parseAltLabel(self, entity, prop, value):
        entity = Symbol(self._parse_name(entity))
        entity_name = Constant(self._parse_name(value))
        x = Symbol.fresh()
        ant = entity(x)
        label = Symbol(self._parse_name(prop))
        con = label(x, entity_name)

        self.parsed_constraints.append(RightImplication(ant, con))
        
    def _parseHasTopConcept(self, entity, prop, value):
        entity = Symbol(self._parse_name(entity))
        entity_name = Constant(self._parse_name(value))
        x = Symbol.fresh()
        ant = entity(x)
        topConcept = Symbol(self._parse_name(prop))
        con = topConcept(x, entity_name)

        self.parsed_constraints.append(RightImplication(ant, con))
        
    def _parseEnumeratedClass(self, entity, prop, value):
        warnings.warn('Not implemented yet: EnumeratedClass')
        
    def _parseEquivalentClass(self, entity, prop, value):
        # No se puede implementar en DL?, <->?
        warnings.warn('Not implemented yet: EquivalentClass')

    def _parseDisjointClass(self, entity, prop, value):
        # inconsistent(C, D) :− disjointWith(C, D), type(X, C), type(X, D).
        warnings.warn('Not implemented yet: DisjointClass')
        
    def _parseProperty(self, entity, prop, value):
        pass
