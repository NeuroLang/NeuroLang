NeuroLang for Logic Programmers
===============================


NeuroLang is implemented on top of Datalog+/- with probabilistic extensions.
The primary way to use it is the Python-embedded frontend described in
:ref:`python_embedded`.

.. _python_embedded:

Using Datalog Embedded in Python
--------------------------------

This requires a first step importing the NeuroLang frontend and initialising the class::

  >>> from neurolang import frontend
  >>> nl = frontend.NeurolangDL()

Then, we can add some facts (connected) and rules (reachable)::

  >>> with nl.environment as e:
  ...   e.connected[0, 1] = True
  ...   e.connected[1, 2] = True
  ...   e.connected[2, 3] = True
  ...   e.reachable[e.x, e.y] = e.connected[e.x, e.y]
  ...   e.reachable[e.x, e.y] = e.reachable[e.x, e.z] & e.connected[e.z, e.y]

please note how the environment ``e`` allows for the creation of logic programming symbols
dynamically.

With this we now have a program loaded in memory which can be explored as::

  >>> print(nl.symbols['connected'])
  connected: typing.AbstractSet[typing.Tuple[int, int]] = (0, 1) ...
  >>> for rule in nl.current_program:
  ...   print(rule)
  reachable(x, y) ← connected(x, y)
  reachable(x, y) ← ( reachable(x, z) ) ∧ ( connected(z, y) )

Finally, we can solve the query::

  >>> with nl.environment as e:
  ...   res = nl.query((e.x, e.y), e.reachable(e.x, e.y))
  >>> print(res)
     x  y
  0  0  1
  1  0  2
  2  0  3
  3  1  2
  4  1  3
  5  2  3

Alternatively the `connected` table can be added more efficiently
from a tuple iterable or a `numpy.array` as follows::

  >>> nl.add_tuple_set([(0, 1), (1, 2), (2, 3)], name='connected')
  connected: typing.AbstractSet[typing.Tuple[int, int]] = (0, 1) ...


Including Aggregations and Builtin Functions
--------------------------------------------

.. note::

   The aggregation API is under active development. The examples below
   illustrate the intended syntax.

Suppose that now we want to obtain the number of destinations
reachable by each starting point. For this we need a new aggregation
function that counts the number of distinct elements. Specifically::

  >>> from typing import Iterable
  >>> from neurolang import frontend
  >>> nl = frontend.NeurolangDL()
  >>> nl.add_tuple_set([(0, 1), (1, 2), (2, 3)], name='connected')
  connected: typing.AbstractSet[typing.Tuple[int, int]] = (0, 1) ...
  >>> @nl.add_symbol
  ... def agg_count(x: Iterable) -> int:
  ...   return len(set(x))
  >>> with nl.environment as e:
  ...   e.reachable[e.x, e.y] = e.connected[e.x, e.y]
  ...   e.reachable[e.x, e.y] = e.reachable[e.x, e.z] & e.connected[e.z, e.y]


This adds a new function `agg_count` to the NeuroLang interpreter,
any built-in or aggregation function is added in the same manner. Then,
we can count all arrivals for starts `0` and `1`::

  >>> with nl.environment as e:
  ...   e.count_arrivals[e.x, agg_count(e.y)] = e.reachable(e.x, e.y)
  ...   counts = nl.query((e.x, e.c), e.count_arrivals(e.x, e.c) & (e.x < 2))
  >>> print(counts)
     x  c
  0  0  3
  1  1  2


Adding Constraints and Open Knowledge Rules
-------------------------------------------

NeuroLang also supports tuple-generating dependencies (TGDs), which are
added with ``add_constraint(antecedent, consequent)``. Variables appearing only
in the consequent are existentially quantified. For instance, we can state
that every person has a parent, even if that parent is not in the data
(:math:`\forall x\, \mathrm{person}(x) \rightarrow \exists y\, \mathrm{parent}(y, x)`)::

  >>> from neurolang.frontend import NeurolangPDL
  >>> nl = NeurolangPDL()
  >>> with nl.environment as e:
  ...   e.parent['John', 'Carl'] = True
  ...   e.parent['Mary', 'Carl'] = True
  ...   e.parent['Pat', 'Anna'] = True
  ...   e.parent['Anna', 'Pete'] = True
  ...   e.person[e.x] = e.parent[e.x, ...]
  ...   e.person[e.x] = e.parent[..., e.x]
  ...   _ = nl.add_constraint(e.person[e.x], e.parent[e.y, e.x])
  ...   e.has_parent[e.x] = e.person[e.x] & e.parent[e.y, e.x]
  ...   res = nl.query((e.x,), e.has_parent(e.x))
  >>> print(res)
        x
  0  Anna
  1  Carl
  2  John
  3  Mary
  4   Pat
  5  Pete

Although only Carl, Anna and Pete have a parent in the data, the constraint
entails that John, Mary and Pat have one too. This holds even when no
``parent`` fact is given at all::

  >>> from neurolang.frontend import NeurolangPDL
  >>> nl = NeurolangPDL()
  >>> with nl.environment as e:
  ...   e.person['Pat'] = True
  ...   e.person['Chris'] = True
  ...   _ = nl.add_constraint(e.person[e.x], e.parent[e.y, e.x])
  ...   e.has_parent[e.x] = e.person[e.x] & e.parent[e.y, e.x]
  ...   res = nl.query((e.x,), e.has_parent(e.x))
  >>> print(res)
         x
  0    Pat
  1  Chris


.. Adding Probabilistic Facts, Choices and Querying Them
.. --------------------------------------------

.. NeuroLang can also handle probabilistic facts, and choices. In the
.. probabilistic facts, each tuple of a certain set is considered an
.. independent random variable that can exist or not in a possible world
.. independently with a given probability. For this we need to use NeuroLang's
.. probabilistic solver. For instance::
  
