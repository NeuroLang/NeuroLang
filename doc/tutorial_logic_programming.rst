NeuroLang for Logic Programmers
===============================


NeuroLang is implemented over the basis of Datalog+/- with probabilistic extensions. In that there are two main frontend which might came useful: the python_embedded_ frontend and the datalog_ frontend

.. python_embedded_

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

please note how the environment :python:`e` allows for the creation of logic programming symbols
dynamically.

With this we now have program loaded in memory which can be explored as::

  >>> print(nl.symbols['connected'])
  connected: typing.AbstractSet[typing.Tuple[int, int]] = [(0, 1), (1, 2), (2, 3)]
  >>> for rule in nl.current_program:
  ...   print(rule)
  reachable(x, y) ← connected(x, y)
  reachable(x, y) ← ( reachable(x, z) ) ∧ ( reachable(z, y) )

Finally, we can solve the query::

  >>> with nl.environment as e:
  ...   res = nl.query((e.x, e.y), e.reachable(e.x, e.y))
  >>> print(res)
     0  1
  0  0  1
  1  1  2
  2  2  3
  3  0  2
  4  1  3
  5  0  3

Alternatively the `connected` table can be added more efficiently
from a tuple iterable or a `numpy.array` as follows::

  >>> nl.add_tuple_set([(0, 1), (1, 2), (2, 3)], name='connected')


Including Aggregations and Builtin Functions
--------------------------------------------

Suppose that now we want to obtain the number of destinations
reachable by each starting point. For this we need a new aggregation
function that counts the number of distinct elements. Specifically::

  >>> from typing import Iterable
  >>> from neurolang import frontend
  >>> nl = frontend.NeurolangDL()
  >>> nl.add_tuple_set([(0, 1), (1, 2), (2, 3)], name='connected')
  >>> @nl.add_symbol
  >>> def agg_count(x: Iterable) -> int:
  >>>   return len(set(x))
  >>> with nl.environment as e:
  ...   e.reachable[e.x, e.y] = e.connected[e.x, e.y]
  ...   e.reachable[e.x, e.y] = e.reachable[e.x, e.z] & e.connected[e.z, e.y]


This adds a new function `agg_count` to the NeuroLang interpreter,
any built-in or aggregation function is added in the same manner. Then,
we can then count all arrivals for starts `0` and `1`::

  >>> with nl.environment as e:
  >>>   e.count_arrivals[e.x, agg_count(e.y)] = e.reachable(e.x, e.y)
  >>>   counts = nl.query((e.x, e.c), e.count_arrivals(e.x, e.c) & (e.x < 2))
  >>> print(counts)
     0  1
  0  0  3
  1  1  2


Adding Constraints and Open Knowledge Rules
-------------------------------------------

Neurolang also supports tuple-generating dependencies (TGDs).
We can say that a person is a parent if they have a child::

  >>> from neurolang.frontend import NeurolangPDL
  >>> nl = NeurolangPDL()
  >>> with nl.environment as e:
  ...   e.parent['John', 'Carl'] = True
  ...   e.parent['Mary', 'Carl'] = True
  ...   e.parent['Pat', 'Anna'] = True
  ...   e.parent['Anna', 'Pete'] = True
  ...   e.person[e.x] = e.parent[e.x, ...]
  ...   e.person[e.x] = e.parent[..., e.x]
  ...   nl.add_constraint(e.person[e.x], e.person[e.y] & e.parent[e.y, e.x])
  ...   e.has_parent[e.x] = e.person[e.x] & e.parent[e.x, e.y]
  >>> res = nl.solve_all()
  >>> print(res)


  >>> from neurolang.frontend import NeurolangPDL
  >>> nl = NeurolangPDL()
  >>> with nl.environment as e:
  ...   e.person['Pat'] = True
  ...   e.person['Chris'] = True
  ...   nl.add_constraint(e.person[e.x], e.parent[e.y, e.x])
  ...   e.has_grand_parent[e.x] = e.person[e.x] & e.person[e.y] & e.parent[e.y, e.x] & e.parent[e.z, e.y]
  >>> res = nl.solve_all()
  >>> print(res)


.. Adding Probabilistic Facts, Choices and Querying Them
.. --------------------------------------------

.. Neurolang can also handle probabilistic facts, and choices. In the
.. probabilistic facts, each tuple of a certain set is considered an
.. independent random variable that can exist or not in a possible world
.. independently with a given probability. For this we need to use Neurolang's
.. probabilistic solver. For instance::
  
