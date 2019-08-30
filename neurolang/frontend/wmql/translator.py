import os
import operator as op

import tatsu
from tatsu.model import NodeWalker, ModelBuilderSemantics

from ...solver_datalog_naive import (
    Symbol, Constant,
    Implication, Unknown,
)
from ...datalog_magic_sets import SymbolAdorned


EBNF_FILE_NAME = os.path.join(os.path.dirname(__file__), 'wmql.ebnf')
with open(EBNF_FILE_NAME) as f:
    WMQL_EBNF = r''.join(f.readlines())


__all__ = ['WMQLDatalogSemantics', 'wmql_grammar']


class WMQLDatalogSemantics(NodeWalker):
    def __init__(
        self,
        tracts_symbol_name='tracts',
        regions_symbol_name='regions',
        tract_traversals_symbol_name='tract_traversals',
        endpoints_in_symbol_name='endpoints_in',
        parser=None, paths=None
    ):
        self.aux_predicate_number = 0
        self.aux_variable_number = 0
        self.aux_implications = []
        self.tracts_symbol_name = tracts_symbol_name
        self.regions_symbol_name = regions_symbol_name
        self.tract_traversals_symbol_name = tract_traversals_symbol_name
        self.endpoints_in_symbol_name = endpoints_in_symbol_name
        self.parser = parser
        if paths is None:
            self.paths = []
        else:
            self.paths = paths
        self.paths.append('.')

    def get_fresh_variable(self, type_=Unknown):
        v = Symbol[type_](f'x_{self.aux_variable_number}')
        self.aux_variable_number += 1
        return v

    def get_fresh_functor(self, tracts_regions=False):
        if not tracts_regions:
            ret = Symbol(f'p_{self.aux_predicate_number}')
        else:
            ret = (
                SymbolAdorned(
                    f'p_{self.aux_predicate_number}', 'tracts', None
                ),
                SymbolAdorned(
                    f'p_{self.aux_predicate_number}', 'regions', None
                )
            )
        self.aux_predicate_number += 1
        return ret

    def walk_blocked_assignment(self, node, **kwargs):
        return self.walk_assignment(node, **kwargs)

    def walk_assignment(self, node, **kwargs):
        ret = []
        x = Symbol('x')
        kwargs['query_var'] = x
        if node.identifier.hemisphere is not None:
            if node.identifier.hemisphere == 'side':
                sides = ('left', 'right')
            else:
                sides = (node.identifier.hemisphere, )
            for side in sides:
                kwargs['side'] = side
                c_tracts, c_regions = self.walk(node.identifier, **kwargs)
                a_tracts, a_regions = self.walk(node.value, **kwargs)
                ret.append(Implication(c_tracts, a_tracts))
                ret.append(Implication(c_regions, a_regions))
        else:
            c_tracts, c_regions = self.walk(node.identifier, **kwargs)
            a_tracts, a_regions = self.walk(node.value, **kwargs)
            ret = [
                Implication(c_tracts, a_tracts),
                Implication(c_regions, a_regions),
            ]

        return ret

    def walk_disjunction(self, disjunction, **kwargs):
        if len(disjunction.term) == 1:
            ret = self.walk(disjunction.term[0], **kwargs)
        else:
            p_tracts, p_regions = self.get_fresh_functor(tracts_regions=True)
            ret = (
                p_tracts(kwargs['query_var']),
                p_regions(kwargs['query_var']),
            )
            x = Symbol('x')
            kwargs_aux = kwargs.copy()
            kwargs_aux['query_var'] = x
            code = kwargs['code']
            head_tracts = p_tracts(x)
            head_regions = p_regions(x)
            for term in disjunction.term:
                term_tracts, term_regions = self.walk(term, **kwargs_aux)
                code.append(Implication(head_tracts, term_tracts))
                code.append(Implication(head_regions, term_regions))
        return ret

    def walk_term(self, term, **kwargs):
        factor_tracts, factor_regions = self.walk(term.factor[0], **kwargs)
        for factor in term.factor[1:]:
            factor_tracts_, factor_regions_ = self.walk(factor, **kwargs)
            factor_tracts = factor_tracts & factor_tracts_
            factor_regions = factor_regions & factor_regions_
        return (factor_tracts, factor_regions)

    def walk_function_evaluation(self, fe, **kwargs):
        p_tracts, p_regions = self.get_fresh_functor(tracts_regions=True)
        x = Symbol('x')
        kwargs_aux = kwargs.copy()
        del kwargs_aux['code']
        kwargs_aux['query_var'] = x
        parameter_tracts, parameter_regions = self.walk(
            fe.argument, **kwargs_aux
        )
        kwargs['code'] += [
            Implication(p_regions(x), parameter_regions),
            Implication(p_tracts(x), parameter_tracts)
        ]

        x = self.get_fresh_variable()
        if fe.name == 'endpoints_in':
            f_name = self.endpoints_in_symbol_name
            r_tracts = (
                Symbol(f_name)(kwargs['query_var'], x)
                & p_regions(x)
            )
            r_regions = p_regions(kwargs['query_var'])
        elif fe.name in (
            'anterior_of', 'posterior_of',
            'superior_of', 'inferior_of'
            'left_of', 'right_of',
        ):
            tract = self.get_fresh_variable()
            region = self.get_fresh_variable()
            region_id = self.get_fresh_variable()
            f_name = f'wmql_{fe.name}'
            r_tracts = (
                Symbol(self.tracts_symbol_name)(kwargs['query_var'], tract) &
                Symbol(self.regions_symbol_name)(region_id, region) &
                p_regions(region_id) &
                Symbol(f_name)(tract, region)
            )
            p_tracts2, _ = self.get_fresh_functor(tracts_regions=True)
            kwargs['code'] += [
                Implication(p_tracts2(kwargs['query_var']), r_tracts)
            ]
            r_tracts = p_tracts2(kwargs['query_var'])
            r_regions = (
                Symbol(
                    self.tract_traversals_symbol_name
                )(x, kwargs['query_var'])
                & p_tracts2(x)
            )
        else:
            raise NotImplementedError()

        return (r_tracts, r_regions)

    def walk_factor(self, factor, **kwargs):
        return self.walk(factor.atom, **kwargs)

    def walk_identifier(self, identifier, **kwargs):
        if identifier.hemisphere is not None:
            if identifier.hemisphere == 'side':
                side = kwargs['side']
            elif identifier.hemisphere == 'opposite':
                if kwargs['side'] == 'left':
                    side = 'right'
                else:
                    side = 'left'
            else:
                side = identifier.hemisphere
            name = f'{identifier.name}_{side}'
        else:
            name = identifier.name

        return (
            SymbolAdorned(name, 'tracts', None)(kwargs['query_var']),
            SymbolAdorned(name, 'regions', None)(kwargs['query_var']),
        )

    def walk_number(self, number, **kwargs):
        return (
            Symbol(self.tract_traversals_symbol_name
                   )(kwargs['query_var'], Constant(int(number.n))),
            Constant(op.eq)(kwargs['query_var'], Constant(int(number.n)))
        )

    def walk_module_import(self, imp, **kwargs):
        import_code = []
        new_kwargs = kwargs.copy()
        del new_kwargs['code']
        for filename in imp.filename:
            for path in self.paths:
                fn = os.path.join(path, filename)
                if os.path.exists(fn):
                    with open(fn) as f:
                        script = f.read()
                        model = self.parser.parse(script.lower())
                        res = self.walk(model, **new_kwargs)
                        import_code += res
                    break
            else:
                print(f'File not found {filename}')
        return import_code

    def walk_list(self, node_list, **kwargs):
        code = kwargs.get('code', [])
        kwargs['code'] = code
        for i, node in enumerate(node_list):
            w = self.walk(node, **kwargs)
            if w is not None:
                code += w
        return code


wmql_grammar = tatsu.compile(
    WMQL_EBNF, name='WMQL', semantics=ModelBuilderSemantics()
)
