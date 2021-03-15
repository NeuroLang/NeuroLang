import io
import xml.etree.ElementTree as ET
from ..logic import Implication, Symbol, Conjunction
from .constraints_representation import RightImplication


#TODO 
# 1) Refactor
# 2) DONE: Change symbol names (x, y, ...) and rewrite asserts in test 
# 3) Solve the temporary trick in __init__ 

class OntologyParser:
    """
    This class is in charge of generating the rules that can be derived
    from an ontology, both at entity and constraint levels.
    """
    OBJECT_PROPERTY = '{http://www.w3.org/2002/07/owl#}ObjectProperty'
    CLASS = '{http://www.w3.org/2002/07/owl#}Class'
    RESTRICTION = '{http://www.w3.org/2002/07/owl#}Restriction'

    def __init__(self, path):

        if isinstance(path, io.StringIO):
            '''This is a temporary trick to be able to use StringIO in the 
                tests. A more elegant solution should be found:
                    - Restart the iterator.
                    - Parse the namespaces from ET.parse(...)
            '''
            data = io.StringIO(path.getvalue())
            self.namespace = dict([node for _, node in ET.iterparse(data, events=['start-ns'])])
        else:
            self.namespace = dict([node for _, node in ET.iterparse(path, events=['start-ns'])])
        onto = ET.parse(path)
        self.root = onto.getroot()

    def parse_ontology(self):
        props = self._parse_object_properties()
        classes, constraints = self._parse_classes()

        return props + classes, constraints
        
    def _parse_object_properties(self):
        props = []
        for obj in self.root.findall(self.OBJECT_PROPERTY):
            prop = self._cut_name_from_item(obj)
            domain = self._cut_name_from_item(obj.find('rdfs:domain', self.namespace))
            rng = self._cut_name_from_item(obj.find('rdfs:range', self.namespace))
            
            if domain is not None and rng is not None:
                x = Symbol.fresh()
                y = Symbol.fresh()
                sym_dom = Symbol(domain)(x)
                sym_rng = Symbol(rng)(y)
                sym_prop = Symbol(prop)(x, y)
                imp_dom = Implication(sym_dom, sym_prop)
                imp_rng = Implication(sym_rng, sym_prop)

                props = props + [imp_dom, imp_rng]

        return props

    def _parse_classes(self):
        classes = []
        constraints = []
        for obj in self.root.findall(self.CLASS):
            imp_label = None
            prop = self._cut_name_from_item(obj)
            label = self._get_name_from_text_atribute(obj.find('rdfs:label', self.namespace))
            if label is not None:
                new_prop = label.replace(' ', '_')
                if new_prop != prop:
                    x = Symbol.fresh()
                    imp_label = Implication(Symbol(prop)(x), Symbol(new_prop)(x))
                    classes.append(imp_label)
            rules, cons = self._parse_subclass_and_restriction(obj.findall('rdfs:subClassOf', self.namespace), prop)
            classes = classes + rules
            constraints = constraints + cons

        return classes, constraints


    def _cut_name_from_item(self, obj):    
        if obj is None:
            return None
        
        items = obj.items()
        if len(items) > 1:
            raise Exception(f'More than 1 element, check: {items}')
        split = items[0][1].split('#')
        if len(split) == 1:
            name = split[0].lower()
        else:
            name = split[1].lower()
            
        return name

    def _get_name_from_text_atribute(self, obj):
        if hasattr(obj, 'text'):
            label = obj.text.lower()
        else:
            label = None

        return label

    def _parse_subclass_and_restriction(self, list_of_subclasses, prop):
        parsed_rules = []
        parsed_constraints = []
        for subClassOf in list_of_subclasses:
            #If the inner class is defined, there is an intersection
            inner_classes = subClassOf.findall('owl:Class', self.namespace)
            
            if not inner_classes:
                #If there is no inner class, there may be a restriction
                restrictions = subClassOf.findall('owl:Restriction', self.namespace)
                if not restrictions:
                    #If there is no inner class and no restriction, 
                    #then there is only a subclass definition.
                    cons = Symbol(self._cut_name_from_item(subClassOf))
                    ant = Symbol(prop)
                    x = Symbol.fresh()
                    imp = Implication(cons(x), ant(x))
                    parsed_rules.append(imp)
                else:
                    for res in restrictions:
                        onProperty = res.find('owl:onProperty', self.namespace)
                        type_of_restriction, names = self._parse_restriction(res)

                        x = Symbol.fresh()
                        y = Symbol.fresh()
                        support_prop = Symbol.fresh()
                        if type_of_restriction == 'someValuesFrom' and names:
                            for value in names:
                                parsed_rules.append(Implication(Symbol(value)(y), support_prop(x, y)))
                            onProp = Symbol(self._cut_name_from_item(onProperty))
                            prop_imp = Implication(onProp(x, y), support_prop(x, y))
                            exists_imp = RightImplication(Symbol(prop)(x), support_prop(x, y))
                        
                            parsed_rules.append(prop_imp)
                            parsed_constraints.append(exists_imp)

                        if type_of_restriction == 'allValuesFrom' and names:
                            onProp = Symbol(self._cut_name_from_item(onProperty))
                            conj = Conjunction((prop(x), onProp(x, y)))
                            for value in names:
                                parsed_rules.append(Implication(Symbol(value)(y), conj))
                        
            else:
                p_rules, p_constraints = self._parse_inner_classes(inner_classes, prop)
                parsed_rules = parsed_rules + p_rules
                parsed_constraints = parsed_constraints + p_constraints
                
        return parsed_rules, parsed_constraints
                    
    def _parse_inner_classes(self, inner_classes, prop):
        x = Symbol.fresh()
        y = Symbol.fresh()
        parsed_rules = []
        parsed_constraints = []
        for _class in inner_classes:
            intersection = _class.findall('owl:intersectionOf', self.namespace)
            if not intersection:
                cons = Symbol(self._cut_name_from_item(_class))
                ant = Symbol(prop)
                imp = Implication(cons(x), ant(x))
                parsed_rules.append(imp)
            else:
                support_prop = Symbol.fresh()
                for inter in intersection:
                    for child in inter.getchildren():

                        if child.tag == self.RESTRICTION:
                            onProperty = child.find('owl:onProperty', self.namespace)
                            type_of_restriction, names = self._parse_restriction(child)
                            
                            if type_of_restriction == 'someValuesFrom' and names:
                                for value in names:
                                    parsed_rules.append(Implication(Symbol(value)(y), support_prop(x, y)))
                                onProp = Symbol(self._cut_name_from_item(onProperty))
                                prop_imp = Implication(onProp(x, y), support_prop(x, y))
                                exists_imp = RightImplication(Symbol(prop)(x), support_prop(x, y))
                            
                                parsed_rules.append(prop_imp)
                                parsed_constraints.append(exists_imp)

                            if type_of_restriction == 'allValuesFrom' and names:
                                onProp = Symbol(self._cut_name_from_item(onProperty))
                                conj = Conjunction((prop(x), onProp(x, y)))
                                for value in names:
                                    parsed_rules.append(Implication(Symbol(value)(y), conj))
                        
                        if child.tag == self.CLASS:
                            cons = Symbol(self._cut_name_from_item(child))
                            imp = Implication(cons(x), support_prop(x, y))
                            parsed_rules.append(imp)

        return parsed_rules, parsed_constraints


    def _parse_restriction(self, subClassOf):
        names = []
        someValuesFrom = subClassOf.find('owl:someValuesFrom', self.namespace)
        if someValuesFrom is not None:
            inner_class = someValuesFrom.findall('owl:Class', self.namespace)
            if inner_class:
                for _class in inner_class:
                    name = self._cut_name_from_item(_class)
                    names.append(name)
            else:
                names = [self._cut_name_from_item(someValuesFrom)]

            return 'someValuesFrom', names

        allValuesFrom = subClassOf.find('owl:allValuesFrom', self.namespace)
        if allValuesFrom is not None:
            inner_class = allValuesFrom.findall('owl:Class', self.namespace)
            if inner_class:
                for _class in inner_class:
                    unionOf = _class.findall('owl:unionOf', self.namespace)
                    if unionOf:
                        for union in unionOf:
                            for u in union.getchildren():
                                name = self._cut_name_from_item(u)
                                names.append(name)
                    else:
                        name = self._cut_name_from_item(_class)
                        names.append(name)
            else:
                names = [self._cut_name_from_item(allValuesFrom)]
                #elements = [inner_class]

            return 'allValuesFrom', names

        if allValuesFrom is None and someValuesFrom is None:
            raise Exception(f'Restriction not parsed: {subClassOf}')         
        
        
