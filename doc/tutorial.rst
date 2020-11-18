
Get Started with NeuroLang
==========================


First Steps With NeuroLang
--------------------------
NeuroLang is a unifying formalism to perform complex queries and explorations using heterogeneous data sources like tabular data,
volumetric images, and ontologies. To perform this in sound manner, NeuroLang is a probabilistic logic programming language based on Datalog [abiteboul1995]_, [maier2018]_. 


The whole idea of logic programming is to be able to make assertions of the style:

  region `x` is a left hemisphere gyrus **if** the label of `x` in Destrieux et al's atlas starts with "L G".

which can be formalised in first order logic as

.. math::
  
   (\forall x) \operatorname{left\_hemisphere\_gyrus}(x) \leftarrow (\exists l) \operatorname{region}(x) \wedge \operatorname{destrieux\_label}(l, x) \wedge \operatorname{startswith}('L\, G', l)


which, if we assume that :math:`x` being on Destrieux et al's atlas means that :math:`x` is already a region, can be shortened as

.. math::
  
   (\forall x) \operatorname{left\_hemisphere\_gyrus}(x) \leftarrow (\exists l)  \operatorname{destrieux\_atlas}(l, x) \wedge \operatorname{startswith}('L\,G', l)


Finally, for notation convenience, we will drop the quantifiers, assuming that all variable on the left of the arrow (such as :math:`x`) is universally quantified, and all variable appearing only on the right of the arrow (such as :math:`l`) will be existentially quantified [maier2018]_. This leads to the expression 


.. math::
  
   \operatorname{left\_hemisphere\_gyrus}(x) \leftarrow  \operatorname{destrieux\_atlas}(l, x) \wedge \operatorname{startswith}('L\,G', l)


which we formalise in python as:

.. code-block:: python

  with neurolang.scope as e:
      e.left_hemisphere_gyrus[e.x] = (
          e.destrieux_atlas(e.l, e.x) &
          e.startswith('L G', e.l)
      )


the full example is in our gallery in :ref:`sphx_glr_auto_examples_plot_load_destrieux_left_hemisphere_gyri.py`.


Negation can also be used in Neurolang. For instance


Disjunctions in Logic Programming
.................................


Disjunctions in logic programming merit are a very specific case. For instance, let's say that all the regions in the left hemisphere's cortex are either a sulcus or gyrus, or more specifically

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


which we formalise in Neurlang in the classical logical programming syntax:

.. code-block:: python

  with neurolang.scope as e:
      e.left_hemisphere_region[e.x] = e.left_hemisphere_sulcus(e.x)
      e.left_hemisphere_region[e.x] = e.left_hemisphere_gyrus(e.x) 



or in a less verbose manner:

.. code-block:: python

  with neurolang.scope as e:
      e.left_hemisphere_region[e.x] = e.left_hemisphere_sulcus(e.x) | e.left_hemisphere_gyrus(e.x)



The Case of Negation
.....................

Negation in logic programming needs special observance. We assume a *close world*
viewpoint of the world. This means that can only predicate on things we know.
As a consequence, if a predicate is negated using the operator :code:`~`, then 
at lest one variable of the negated predicate needs to be present in a positive predicate. 
Take into account the following query

.. code-block:: python

  @neurolang.add_symbol
  def even(x: int) -> bool:
      return x % 2 == 0

  with neurolang.scope as e:
      e.odd[e.x] = ~e.even[e.x]

      res = neurolang.query(e.x, e.odd(e.x))


is not a *valid* logic program as we don't know anyhing *positive* about `e.x`.
However if we restrict the domain of `x` as being between one and ten:

.. code-block:: python

  between_one_and_ten = nl.add_tuple_set([(x,) for x in range(1, 10)])

  with neurolang.scope as e:
      e.odd[e.x] = ~e.even[e.x]

      res = neurolang.query(
          e.x, 
          between_one_and_ten(e.x) & e.odd(e.x)
      )

we will obtain the set of tuples :code:`{(1,), (3,), (5,), (7,), (9,)}`

Aggregations
............

Aggregations combine information from a set of tuples. A good example of an aggregation is the maximum. As a mathematical definition we could define an aggregation as

.. math::

  \begin{split}
  (\forall country)\operatorname{max\_population\_per\_country}\left(country, max(\{pop: (\exists province)\operatorname{population\_per\_country\_province}(country, province, pop)\})\right) \leftarrow \\
  (\exists province)(\exists pop)\operatorname{population\_per\_country\_province}(country, province, pop)
  \end{split}

which in neurolang is expressed as

.. code-block:: python

   with neurolang.scope as e:
       e.max_population_per_country[e.country, e.max(e.pop)] = e.population_per_country_province(e.country, e.province, e.pop)


Syntactic Sugar
...............


Some syntactic sugar has been included to make queries easier to write.

Use First Column to Reference Second in a Set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a set has only two columns you can use the first to identify elements in
the second one. For instance if you have a set with regions in an altas


.. list-table:: Destrieux
   :widths: 5 5
   :header-rows: 1

   * - Region Name
     - Region
   * - "Central Sulcus"
     - R1


Then the following query

.. code-block:: python
   
    with neurolang.scope as e:
        e.central[e.x] = Destrieux.s["Central Sulcus"] == e.x


Is equivalent to 

.. code-block:: python
   
    with neurolang.scope as e:
        e.central[e.x] = Destrieux["Central Sulcus", e.x]

for an example of application see :ref:`sphx_glr_auto_examples_plot_sulcal_queries.py` .


Logical Quantifiers
~~~~~~~~~~~~~~~~~~~

Assume that you want the most anterior region in Destrieux et al's atlas. Then,
a way to phrase it is

 The most anterior region `r` is such that `r` belongs to Destrieux's atlas and
 no existing region `q` in Destrieux's atlas is different from `r` and anterior to it.

we can formalise it as

.. code-block :: python

    with neurolang.scope as e:
        e.most_anterior[e.r] = (
            Destrieux[..., e.r] &
            ~neurolang.exists(
                e.q,
                Destrieux(..., e.q) & (e.r != e.q) &
                e.anterior_of(e.q, e.r)
            )
        )
  

Probabilistic Semantics in Neurolang
------------------------------------

One of the main features of NeuroLang are its probabilistic semantics. Specifically, NeuroLang can represent discrete random variables and execute inference tasks.
For a more detailed description of probabilistic logic languages please see the introduction of De Raedt et al [deraedt2015]_. Let's present an example:

In neuroimaging meta-analysis, a main question is the probability that an article containg a specific term, such as "auditory" reports a voxel in the brain as active.
To model this consider a set of articles as independent identically distributed samples of *neuroscientific knowledge*. Then we also consider known the probability that
in a given article, a voxel is reported as active, and also a term is reported as present. Both with a certain probability. Our goal is then to compute, as done by Yarkoni et al. [yarkoni2011]_,
the probability that a voxel is reported as active, when a word is on the study. We can formalise this problem mathematically as:

.. math::

    \mathbf{s} &\sim \operatorname{Choice}\left(\{S_1, \ldots, S_n\}, \frac 1 n \right) \\
    \mathbf{t_1}, \mathbf{s} &\sim \operatorname{Bernoulli}\left(\theta_{t_1, \mathbf{s}}\right)\\
    &\vdots\\
    \mathbf{t_T}, \mathbf{s} &\sim \operatorname{Bernoulli}\left(\theta_{t_T, \mathbf{s}}\right)\\
    \mathbf{v_1}, \mathbf{s} &\sim \operatorname{Bernoulli}\left(\theta_{v_1, \mathbf{s}}\right)\\
    &\vdots\\
    \mathbf{v_V}, \mathbf{s} &\sim \operatorname{Bernoulli}\left(\theta_{v_V, \mathbf{s}}\right)\\

where :math:`\mathbf{s}` is a chosen study amongs those in the set :math:`S_1,\ldots,S_n`; :math:`\mathbf{t_i},\mathbf{s}` is the probability that the i-th term  is present in study *s*;
and :math:`\mathbf{v_j}, \mathbf{s}` is the joint probability that the j-th voxel is reported active in study *s*. Our goal then is to compute the probability, that a voxel `j` is active
if a certain term :math:`t_i` is reported in the study :math:`P(\mathbf{v_j}|\mathbf{t_i})`. In the following NeuroLang snippet we setup the computation of this conditional probability as well as
of the marginal probability that a term has been mentioned in a study:

.. code:: python

  with neurolang.scope as e:
      # Express the marginal random variables for term and voxel
      e.TermAssociation[e.t] = e.SelectedStudy[e.s] & e.TermInStudy[e.t, e.s]
      e.Voxel[e.v] = e.SelectedStudy[e.s] & e.VoxelReported[e.v, e.s]

      # Compute the marginal probability of a term with respect to the program
      e.term_marginal_probability[e.t, e.PROB[e.t]] = e.TermAssociation[a.t]

      # compute the probability of a voxel being present conditioned to that of the term
      # auditory being present.
      e.probmap[e.v, e.PROB[e.v]] = e.Voxel[e.v] // e.TermAssociation["auditory"] 

      #perform the calculations
      term_marginal_probability = nl.query((e.t, e.prob), e.term_marginal_probability(e.t, e.prob))
      voxel_conditional_probability = nl.query((e.v, e.prob), e.probmap(e.v, e.prob))

For a more complete example including the prodiduction of the following image,
see the example :ref:`sphx_glr_auto_examples_plot_neurosynth_implementation.py`.

.. figure::  /auto_examples/images/thumb/sphx_glr_plot_neurosynth_implementation_thumb.png
          :scale: 100%
          :align: center

          Probability that a voxel is active is "auditory" is mentioned in a study. Only the voxels with
          a probability within the largest 5% are shown.


.. [abiteboul1995] Abiteboul, S., Hull, R. & Vianu, V. Foundations of databases. (Addison Wesley, 1995).
.. [deraedt2015] De Raedt, L. & Kimmig, A. Probabilistic (logic) programming concepts. Mach Learn 100, 5–47 (2015).
.. [maier2018] Maier, D., Tekle, K. T., Kifer, M. & Warren, D. S. Datalog: concepts, history, and outlook. in Declarative Logic Programming (eds. Kifer, M. & Liu, Y. A.) 3–100 (Association for Computing Machinery and Morgan & Claypool, 2018). doi:10.1145/3191315.3191317.
.. [yarkoni2011] Yarkoni, T., Poldrack, R. A., Nichols, T. E., Van Essen, D. C. & Wager, T. D. Large-scale automated synthesis of human functional neuroimaging data. Nat Meth 8, 665–670 (2011).

