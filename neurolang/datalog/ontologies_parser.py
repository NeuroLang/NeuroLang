import rdflib
import pandas as pd
import nibabel as nib
from nilearn import datasets
from .ontologies_rewriter import RightImplication
from ..expressions import Symbol, ExpressionBlock, Constant
from rdflib import OWL, RDF, BNode

C_ = Constant
S_ = Symbol
EB_ = ExpressionBlock
RI_ = RightImplication


class OntologiesParser():
    def __init__(self, paths, load_format='xml'):
        self.namespaces_dic = None
        self.owl_dic = None
        if isinstance(paths, list):
            self._load_ontology(paths, load_format)
        else:
            self._load_ontology([paths], load_format)

    def _load_ontology(self, paths, load_format):
        self._create_graph(paths, load_format)
        self._process_properties()

    def _create_graph(self, paths, load_format):
        g = rdflib.Graph()
        for path in paths:
            g.load(path, format=load_format)

        self.graph = g

    def _process_properties(self):
        predicates = set(self.graph.predicates())
        properties = list(
            map(
                lambda a: list(
                    map(
                        lambda s: s.replace('-', '_'),
                        a.split('/')[-1].split('#')
                    )
                ), predicates
            )
        )
        properties = list(
            map(
                lambda x: x[0] + '_' + self._replace_property(x[1]), properties
            )
        )
        self.predicates_translation = dict(zip(predicates, properties))

    def _replace_property(self, prop):
        if prop in [
            'rdf_schema_subClassOf',
            'rdf_schema_subPropertyOf',
        ]:
            prop = prop + '2'

        return prop

    def parse_ontology(self, neurolangDL):
        self.eb = EB_(())
        self.neurolangDL = neurolangDL

        self._load_domain()
        self._load_properties()
        self._load_constraints()

        self.neurolangDL.load_constraints(self.eb)
        return self.neurolangDL

    def _load_domain(self):
        triple = S_('triple')
        triples = tuple([
            triple(C_(e1), C_(e2), C_(e3)) for e1, e2, e3 in self.get_triples()
        ])

        pointers = list(
            filter(lambda x: isinstance(x, BNode), set(self.graph.subjects()))
        )
        pointer = S_('pointer')
        pointer_list = tuple([pointer(C_(e)) for e in pointers])

        dom = S_('dom')
        x = S_('x')
        y = S_('y')
        z = S_('z')

        dom1 = RightImplication(triple(x, y, z), dom(x))
        dom2 = RightImplication(triple(x, y, z), dom(y))
        dom3 = RightImplication(triple(x, y, z), dom(z))

        self.eb = EB_(
            self.eb.expressions + triples + pointer_list + (dom1, dom2, dom3)
        )

    def _load_properties(self):
        x = S_('x')
        z = S_('z')
        triple = S_('triple')

        symbols = ()
        for _, trans in self.predicates_translation:
            symbol_name = trans
            symbol = S_(symbol_name)
            const = C_(symbol_name)
            symbols += (RightImplication(triple(x, const, z), symbol(x, z)), )

        self.eb = ExpressionBlock(self.eb.expressions + symbols)

        self._parse_subproperties()
        self._parse_subclasses()
        self._parse_disjoint()

    def _parse_subproperties(self):
        rdf_schema_subPropertyOf = S_('rdf_schema_subPropertyOf')
        rdf_schema_subPropertyOf2 = S_('rdf_schema_subPropertyOf2')
        w = S_('w')
        x = S_('x')
        y = S_('y')
        z = S_('z')

        subProperty = RI_(
            rdf_schema_subPropertyOf2(x, y), rdf_schema_subPropertyOf(x, y)
        )
        subProperty2 = RI_(
            rdf_schema_subPropertyOf2(x, y) & rdf_schema_subPropertyOf(y, z),
            rdf_schema_subPropertyOf(x, z)
        )

        owl_inverseOf = S_('owl_inverseOf')
        inverseOf = RI_(
            rdf_schema_subPropertyOf(x, y) & owl_inverseOf(w, x) &
            owl_inverseOf(z, y), rdf_schema_subPropertyOf(w, z)
        )

        rdf_syntax_ns_type = S_('rdf_syntax_ns_type')
        objectProperty = RI_(
            rdf_syntax_ns_type(
                x, C_('http://www.w3.org/2002/07/owl#ObjectProperty')
            ), rdf_schema_subPropertyOf(x, x)
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (
                subProperty,
                subProperty2,
                inverseOf,
                objectProperty,
            )
        )

    def _parse_subclasses(self):
        rdf_schema_subClassOf = S_('rdf_schema_subClassOf')
        rdf_schema_subClassOf2 = S_('rdf_schema_subClassOf2')
        w = S_('w')
        x = S_('x')
        y = S_('y')
        z = S_('z')

        subClass = RI_(
            rdf_schema_subClassOf2(x, y), rdf_schema_subClassOf(x, y)
        )
        subClass2 = RI_(
            rdf_schema_subClassOf2(x, y) & rdf_schema_subClassOf(y, z),
            rdf_schema_subClassOf(x, z)
        )

        rdf_syntax_ns_rest = S_('rdf_syntax_ns_rest')
        ns_rest = RI_(
            rdf_schema_subClassOf(x, y) & rdf_syntax_ns_rest(w, x) &
            rdf_syntax_ns_rest(z, y), rdf_schema_subClassOf(w, z)
        )

        rdf_syntax_ns_type = S_('rdf_syntax_ns_type')
        class_sim = RI_(
            rdf_syntax_ns_type(x, C_('http://www.w3.org/2002/07/owl#Class')),
            rdf_schema_subClassOf(x, x)
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (
                subClass,
                subClass2,
                ns_rest,
                class_sim,
            )
        )

    def _parse_disjoint(self):
        w = Symbol('w')
        x = Symbol('x')
        y = Symbol('y')
        z = Symbol('z')

        owl_disjointWith = S_('owl_disjointWith')
        rdf_schema_subClassOf = S_('rdf_schema_subClassOf')
        disjoint = RI_(
            owl_disjointWith(x, y) & rdf_schema_subClassOf(w, x) &
            rdf_schema_subClassOf(z, y), owl_disjointWith(w, z)
        )

        self.eb = ExpressionBlock(self.eb.expressions + (disjoint, ))

    '''def _parse_somevalue_properties(self):
        w = S_('w')
        x = S_('x')
        y = S_('y')
        z = S_('z')

        owl_onProperty = S_('owl_onProperty')
        owl_onProperty2 = S_('owl_onProperty2')
        onProperty = RI_(owl_onProperty2(x, y), owl_onProperty(x, y))

        owl_someValuesFrom = S_('owl_someValuesFrom')
        owl_someValuesFrom2 = S_('owl_someValuesFrom2')
        someValueFrom = RI_(
            owl_someValuesFrom2(x, y), owl_someValuesFrom(x, y)
        )

        pointer = S_('pointer')
        rdf_schema_subClassOf = S_('rdf_schema_subClassOf')

        temp_triple = RI_(
            pointer(w) & owl_someValuesFrom(w, z) & owl_onProperty(w, y) &
            rdf_schema_subClassOf(x, w), y(x, z)
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (
                onProperty,
                someValueFrom,
                temp_triple,
            )
        )'''

    def _load_constraints(self):
        restriction_ids = []
        for s, _, _ in self.graph.triples((None, None, OWL.Restriction)):
            restriction_ids.append(s)

        for rest in restriction_ids:
            cutted_graph = list(self.graph.triples((rest, None, None)))
            res_type = self._identify_restriction_type(cutted_graph)

            if res_type == 'hasValue':
                self._process_hasValue()
            elif res_type == 'minCardinality':
                self._process_minCardinality()
            elif res_type == 'allValuesFrom':
                self._process_allValuesFrom()
            else:
                pass
                #TODO WARNING

    def _identify_restriction_type(self, list_of_triples):
        for triple in list_of_triples:
            if triple[1] == OWL.onProperty or triple[1] == RDF.type:
                continue
            else:
                return triple[1].rsplit('#')[-1]

        return ''

    def _process_hasValue():
        pass

    def _process_minCardinality():
        pass

    def _process_allValuesFrom():
        pass

    def get_triples(self):
        return self.graph.triples((None, None, None))
