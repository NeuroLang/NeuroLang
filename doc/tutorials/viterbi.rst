Viterbi Decoding with the Max-Product Semiring
===============================================

This tutorial shows how to use the **max-product semiring** and **Viterbi
decoder** to find the most likely state sequence in a hidden Markov model
(HMM) expressed as a CP-Logic program.

Background
----------

In a conventional probability semiring, existential variables are aggregated
by summing over all possible values (``Σ p``).  The **max-product semiring**
replaces sum with **max**, turning the computation into a Viterbi-style
maximum over paths:

.. math::

   \text{score}_t(s) = \max_{s'} \bigl[ \text{score}_{t-1}(s') \times
   \text{transition}(s', s) \bigr] \times \text{emission}_t(s)

The decoder module (:mod:`neurolang.probabilistic.viterbi`) adds
**backpointer tracking** so you can recover *which* previous state achieved
the max — and therefore reconstruct the full most-likely path.


Workflow Overview
-----------------

#. Set up a CP-Logic program with probabilistic facts.
#. Define step-by-step rules for each time step.
#. Solve each step with :class:`~neurolang.probabilistic.MaxProductSemiring`.
#. Recover argmax with :func:`~neurolang.probabilistic.viterbi.compute_backpointers`.
#. Reconstruct the most likely sequence with
   :func:`~neurolang.probabilistic.viterbi.trace_path`.


Example: 2-State, 2-Step HMM
-----------------------------

We model an HMM with two states (``s1``, ``s2``) and two time steps.
The goal is to find the most likely state sequence given the observations.

Imports
~~~~~~~

.. code-block:: python

   from neurolang.datalog import Fact
   from neurolang.expressions import Constant, Symbol
   from neurolang.logic import Conjunction, Implication, Union
   from neurolang.probabilistic import dalvi_suciu_lift
   from neurolang.probabilistic.cplogic.program import CPLogicProgram
   from neurolang.probabilistic.semiring import MaxProductSemiring, ProbabilitySemiring
   from neurolang.probabilistic.viterbi import (
       compute_backpointers, decode_viterbi, trace_path,
   )

   ans = Symbol("ans")
   x = Symbol("x")
   y = Symbol("y")
   z = Symbol("z")


Set up the program
~~~~~~~~~~~~~~~~~~

Probabilistic facts for initial probabilities, transition probabilities,
and emission probabilities:

.. code-block:: python

   cpl = CPLogicProgram()
   init = Symbol("init")
   trans = Symbol("trans")
   emit1 = Symbol("emit1")
   emit2 = Symbol("emit2")

   cpl.add_probabilistic_facts_from_tuples(
       init, {(0.6, "s1"), (0.4, "s2")}
   )
   cpl.add_probabilistic_facts_from_tuples(
       trans,
       {
           (0.8, "s1", "s1"),
           (0.2, "s1", "s2"),
           (0.3, "s2", "s1"),
           (0.7, "s2", "s2"),
       },
   )
   cpl.add_probabilistic_facts_from_tuples(
       emit1, {(0.7, "s1"), (0.2, "s2")}
   )
   cpl.add_probabilistic_facts_from_tuples(
       emit2, {(0.1, "s1"), (0.8, "s2")}
   )


Step 1 (t=1)
~~~~~~~~~~~~

.. code-block:: python

   v1 = Symbol("viterbi_1")
   cpl.walk(Union((
       Implication(v1(x), Conjunction((init(x), emit1(x)))),
   )))

   query1 = Implication(ans(x), v1(x))
   result1 = dalvi_suciu_lift.solve_succ_query(
       query1, cpl, semiring=ProbabilitySemiring()
   )

   # result1:
   #   (0.42, s1)   ← 0.6 × 0.7
   #   (0.08, s2)   ← 0.4 × 0.2


Step 2 (t=2) — forward pass with max-product
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ans2 = Symbol("ans_2")
   cpl.walk(Union((
       Implication(ans2(z), Conjunction((
           v1(y), trans(y, z), emit2(z)
       ))),
   )))

   query2 = Implication(ans(z), ans2(z))
   result2 = dalvi_suciu_lift.solve_succ_query(
       query2, cpl, semiring=MaxProductSemiring()
   )

   # result2:
   #   (0.0336, s1)  ← max(0.42×0.8, 0.08×0.3) × 0.1
   #   (0.0672, s2)  ← max(0.42×0.2, 0.08×0.7) × 0.8

The existential variable ``y`` (previous state) is projected away using
**max** instead of sum, implementing the Viterbi max over paths.


Backpointer extraction
~~~~~~~~~~~~~~~~~~~~~~

To recover which previous state ``y`` achieved the max for each current
state ``z``:

.. code-block:: python

   bp_result = compute_backpointers(cpl, prev_symbol=v1, transition_symbol=trans)

   # bp_result (all (y, z) combinations with scores):
   #   (0.336, s1, s1)   ← v1(s1) × trans(s1, s1)
   #   (0.024, s1, s2)   ← v1(s2) × trans(s2, s1)
   #   (0.084, s2, s1)   ← v1(s1) × trans(s1, s2)
   #   (0.056, s2, s2)   ← v1(s2) × trans(s2, s2)

   # Select the max per next state:
   bp_map = {}
   for row in bp_result.relation.value:
       prob = float(row[0])
       state = str(row[1])
       prev = str(row[2])
       if state not in bp_map or prob > bp_map[state][0]:
           bp_map[state] = (prob, prev)

   # bp_map:
   #   s1 → (0.336, s1)   ← highest: v1(s1) × trans(s1, s1)
   #   s2 → (0.084, s1)   ← highest: v1(s1) × trans(s1, s2)

State ``s2`` traces back to ``s1``: the max for ``s2`` at step 2
(0.084) came from previous state ``s1``, not ``s2`` (0.056).


Path reconstruction
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   path = trace_path({2: {"s1": "s1", "s2": "s1"}}, final_state="s2", steps=[1, 2])
   print(path)   # ['s1', 's2']

The most likely sequence starts at ``s1`` and transitions to ``s2``,
matching the intuition that ``s2`` has the higher emission probability
at step 2 (0.8 vs 0.1) and that both states most likely started from
``s1``.


Using decode_viterbi
--------------------

The :func:`~neurolang.probabilistic.viterbi.decode_viterbi` convenience
function automates the multi-step workflow:

.. code-block:: python

   results, traces, backpointers = decode_viterbi(
       cpl,
       variables={"step": [1, 2]},
       rules={
           1: Implication(v1(x), Conjunction((init(x), emit1(x)))),
           2: Implication(ans2(z), Conjunction((
               v1(y), trans(y, z), emit2(z)
           ))),
       },
       query_symbols={1: v1, 2: ans2},
       transition_symbols={
           2: {"prev": v1, "transition": trans},
       },
   )

   # Forward results
   step2_result = results[2]

   # Backpointers
   path = trace_path(backpointers, final_state="s2", steps=[1, 2])
   print(path)   # ['s1', 's2']

``decode_viterbi`` returns a three-tuple:

``results``
  Forward-pass results as :class:`~neurolang.relational_algebra_provenance.ProvenanceAlgebraSet`
  for each step.

``traces``
  Backpointer results (the full join of previous-step result with the
  transition relation) for steps that have transition info.

``backpointers``
  Dictionary mapping step → ``{state: prev_state}``, ready for
  :func:`~neurolang.probabilistic.viterbi.trace_path`.


API Reference
-------------

.. autofunction:: neurolang.probabilistic.viterbi.compute_backpointers
.. autofunction:: neurolang.probabilistic.viterbi.decode_viterbi
.. autofunction:: neurolang.probabilistic.viterbi.trace_path
