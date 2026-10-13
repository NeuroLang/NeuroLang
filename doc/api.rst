.. _api_ref:

.. currentmodule:: neurolang

API Reference
=============

Complete API documentation for all public NeuroLang modules.
Generated automatically from docstrings.

Most users only need the two frontend classes below:
:class:`NeurolangDL` for deterministic Datalog and :class:`NeurolangPDL`
for probabilistic Datalog. Their most used methods are
``add_tuple_set``, ``add_atlas_set``, ``add_symbol``,
``add_probabilistic_facts_from_tuples``,
``add_probabilistic_choice_from_tuples``,
``add_uniform_probabilistic_choice_over_set``,
``execute_datalog_program``, ``execute_squall_program``, ``query``
and ``solve_all``.


End-User Frontend
-----------------

.. autosummary::
   :toctree: gen_api
   :nosignatures:

    NeurolangDL
    NeurolangPDL

Intermediate Representation Infrastructure
------------------------------------------

.. autosummary::
   :toctree: gen_api
  
   expressions
   expression_pattern_matching
   expression_walker
   exceptions


Intermediate Language Representations
-------------------------------------

.. autosummary::
   :toctree: gen_api
    
   logic
   datalog
   probabilistic
   relational_algebra
   relational_algebra_provenance
   provenance_semiring


Command-line Interface & Engine Registry
-----------------------------------------

.. autosummary::
   :toctree: gen_api

   utils.cli
   utils.engine_registry
   utils.engines.base


Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
