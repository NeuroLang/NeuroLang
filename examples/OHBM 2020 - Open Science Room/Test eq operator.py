# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: neurolang
#     language: python
#     name: neurolang
# ---

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

# +
from neurolang.frontend.probabilistic_frontend import ProbabilisticFrontend
from neurolang import frontend as fe

nl = ProbabilisticFrontend()


# +
nl.add_tuple_set([('Val1', 'var'), ('Val2', 'var'), ('Val3', 'var')], name='test_var')

@nl.add_symbol
def word_lower(name: str) -> str:
    print(name)
    return str(name).lower()


# -

with nl.scope as e:
    e.lower[e.lower] = (
        e.test_var[e.name, 'var'] &
        (e.lower == word_lower(e.name))
    )
    
    nl_results = nl.solve_query(e.ontology_terms[e.name])

nl_results['lower']


