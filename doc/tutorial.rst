
Get Started with NeuroLang
==========================


First Steps With NeuroLang
--------------------------
NeuroLang is a unifying formalism to perform complex queries and explorations using heterogeneous data sources like tabular data,
volumetric images, and ontologies. To perform this in a sound manner, NeuroLang is a probabilistic logic programming language based on Datalog [abiteboul1995]_, [maier2018]_.

The examples of this page run in the REPL, in the browser or in a terminal
(:doc:`repl`), on the ``neurosynth`` engine: type them at the prompt, or
follow *Run in the browser REPL* under each one. They use the Destrieux
atlas of the engine, ``destrieux(label, region)``, whose labels are written
``"L G front sup"`` (left superior frontal gyrus) or ``"R S central"`` (right
central sulcus).

.. note::

   To write the same rules in Python, see :doc:`tutorial_python`.


The whole idea of logic programming is to be able to make assertions of the style:

  region x is a left hemisphere gyrus **if** the label of x in Destrieux et al's atlas starts with "L G".

which can be formalised in first order logic as

.. math::
  
   (\forall x) \operatorname{left\_hemisphere\_gyrus}(x) \leftarrow (\exists l) \operatorname{region}(x) \wedge \operatorname{destrieux\_label}(l, x) \wedge \operatorname{startswith}('L\, G', l)


which, if we assume that :math:`x` being on Destrieux et al's atlas means that :math:`x` is already a region, can be shortened as

.. math::
  
   (\forall x) \operatorname{left\_hemisphere\_gyrus}(x) \leftarrow (\exists l)  \operatorname{destrieux\_atlas}(l, x) \wedge \operatorname{startswith}('L\,G', l)


Finally, for notation convenience, we will drop the quantifiers, assuming that all variables on the left of the arrow (such as :math:`x`) are universally quantified, and all variables appearing only on the right of the arrow (such as :math:`l`) will be existentially quantified [maier2018]_. This leads to the expression 


.. math::
  
   \operatorname{left\_hemisphere\_gyrus}(x) \leftarrow  \operatorname{destrieux\_atlas}(l, x) \wedge \operatorname{startswith}('L\,G', l)


which is written in NeuroLang, ``:-`` standing for the arrow and ``&`` for
the conjunction, and keeping the label rather than the region:

.. nl-repl::

   left_hemisphere_gyrus(l) :- destrieux(l, x) & startswith("L G", l)

The REPL answers ``defined left_hemisphere_gyrus/1``: the rule is in the
session. A query, whose head is ``ans``, asks for its rows:

.. nl-repl::
   :norun:

   ans(l) :- left_hemisphere_gyrus(l)

.. code-block:: text

   ╭────────────────────────────╮
   │ l                          │
   ├────────────────────────────┤
   │ L G and S frontomargin     │
   │ L G and S occipital inf    │
   │ L G and S paracentral      │
   │ …                          │
   ╰────────────────────────────╯
   38 rows · showing 1–20 · .more for next


Negation can also be used in NeuroLang, through the ``~`` operator. For
instance, the labels of the atlas which are not left hemisphere gyri are

.. nl-repl::
   :norun:

   not_left_hemisphere_gyrus(l) :- destrieux(l, x) & ~left_hemisphere_gyrus(l)

Note that every variable of a negated atom must also appear in a positive atom
of the same rule (here ``l`` is bound by ``destrieux``).


Disjunctions in Logic Programming
.................................


Disjunctions in logic programming merit a specific discussion. For instance, let's say that all the regions in the left hemisphere's cortex are either a sulcus or gyrus, or more specifically

   x is a left hemisphere region **if** x is left sulcus **or** x is left gyrus

which in first order logic can be formalised as

.. math::

  (\forall x)\operatorname{left\_hemisphere\_region}(x) \leftarrow \operatorname{left\_hemisphere\_sulcus}(x) \vee \operatorname{left\_hemisphere\_gyrus}(x)


alternatively, this can be written as a set of two propositions

.. math::

 \begin{cases}
  (\forall x)\operatorname{left\_hemisphere\_region}(x) \leftarrow \operatorname{left\_hemisphere\_sulcus}(x)\\
  (\forall x)\operatorname{left\_hemisphere\_region}(x) \leftarrow  \operatorname{left\_hemisphere\_gyrus}(x)
 \end{cases}


which we write in NeuroLang as two rules of the same head, with the left
sulci defined as the gyri were:

.. nl-repl::
   :norun:

   left_hemisphere_sulcus(l) :- destrieux(l, x) & startswith("L S", l)
   left_hemisphere_region(l) :- left_hemisphere_sulcus(l)
   left_hemisphere_region(l) :- left_hemisphere_gyrus(l)

or in a less verbose manner, with ``|``:

.. nl-repl::
   :norun:

   left_hemisphere_region(l) :- left_hemisphere_sulcus(l) | left_hemisphere_gyrus(l)

The second form defines the same relation; typed after the first, it adds
rules of the same meaning, which change nothing. ``.idb`` lists the rules of
the session, and ``.drop left_hemisphere_region`` removes them.


Aggregations
............

Aggregations combine information from a set of tuples. A good example of an aggregation is the maximum. As a mathematical definition we could define an aggregation as

.. math::

  \begin{split}
  (\forall country)\operatorname{max\_population\_per\_country}\left(country, max(\{pop: (\exists province)\operatorname{population\_per\_country\_province}(country, province, pop)\})\right) \leftarrow \\
  (\exists province)(\exists pop)\operatorname{population\_per\_country\_province}(country, province, pop)
  \end{split}

which in NeuroLang is expressed as a rule whose head applies the aggregation
to a variable of its body:

.. code-block:: none

   max_population_per_country(country, max(pop)) :- population_per_country_province(country, province, pop)

On the engine, ``destrieux_activated(label, study)`` holds the regions that
the peaks of each study fall in; the number of studies activating each region
is a count:

.. nl-repl::

   studies_per_region(l, count(s)) :- destrieux_activated(l, s)
   ans(l, n) :- studies_per_region(l, n)

and ``.sort n:desc`` puts the regions most often activated first:

.. code-block:: text

   ╭─────────────────────────────┬──────╮
   │ l                           │ n    │
   ├─────────────────────────────┼──────┤
   │ L G front sup               │ 6939 │
   │ R G front sup               │ 6389 │
   │ R G front middle            │ 4953 │
   │ L G front middle            │ 4892 │
   │ …                           │      │
   ╰─────────────────────────────┴──────╯
   148 rows · showing 1–20 · .more for next

The next steps are :doc:`tutorial_datalog`, the whole language through a
complete analysis of the literature, and :doc:`repl`, the REPL itself.

.. [abiteboul1995] Abiteboul, S., Hull, R. & Vianu, V. Foundations of databases. (Addison Wesley, 1995).
.. [maier2018] Maier, D., Tekle, K. T., Kifer, M. & Warren, D. S. Datalog: concepts, history, and outlook. in Declarative Logic Programming (eds. Kifer, M. & Liu, Y. A.) 3–100 (Association for Computing Machinery and Morgan & Claypool, 2018). doi:10.1145/3191315.3191317.

