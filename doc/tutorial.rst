
Get Started with NeuroLang
==========================


First Steps With NeuroLang
--------------------------
NeuroLang is a unifying formalism to perform complex queries and explorations using heterogeneous data sources like tabular data,
volumetric images, and ontologies. To perform this in sound manner, NeuroLang is a probabilistic logic programming language based on Datalog [abiteboul1995]_, [maier2018]_. 


The whole idea of logic programming is to be able to make assertions of the style:

  region x is a left hemisphere gyrus if the label of x in Destrieux et al's atlas starts with "L G".

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
      e.left_hemisphere_gyrus[e.x] = e.destrieux_atlas(e.l, e.x) & e.startswith('L G', e.l)


the full example is in our gallery in :ref:`sphx_glr_auto_examples_plot_load_destrieux_left_hemisphere_gyri.py`.


.. [abiteboul1995] Abiteboul, S., Hull, R. & Vianu, V. Foundations of databases. (Addison Wesley, 1995).
.. [maier2018] Maier, D., Tekle, K. T., Kifer, M. & Warren, D. S. Datalog: concepts, history, and outlook. in Declarative Logic Programming (eds. Kifer, M. & Liu, Y. A.) 3–100 (Association for Computing Machinery and Morgan & Claypool, 2018). doi:10.1145/3191315.3191317.

