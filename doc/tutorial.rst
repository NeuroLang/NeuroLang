
Get Started with NeuroLang
==========================


First Steps With NeuroLang
--------------------------
NeuroLang is a unifying formalism to perform complex queries and explorations using heterogeneous data sources like tabular data,
volumetric images, and ontologies. To perform this in sound manner, NeuroLang is a probabilistic logic programming language based on Datalog [abiteboul1995]_, [maier2018]_. A simple example can be produced analysing Destrieux et al's atlas [destrieux2010]_ and selecting all regions anterior to the central sulcus and superior to the superior temporal sulcus. In terms of logic this can be written as:

.. math::
   \forall name: \forall sulcus: query(name, sulcus) \leftarrow 
    destrieux(name, sulcus) \wedge 
    \exists sts: \exists cs:(destrieux(\text{'superior temporal sulcus'}, sts) \wedge destrieux(\text{'central sulcus'}, cs) \wedge anterior(s, cs) \wedge superior(s, sts))


which is formalised in the Neurolang Python embedded frontend as

.. literalinclude:: ../examples/plot_load_destrieux.py
   :language: python
   

.. [abiteboul1995] Abiteboul, S., Hull, R. & Vianu, V. Foundations of databases. (Addison Wesley, 1995).
.. [destrieux2010] Destrieux, C., Fischl, B., Dale, A. & Halgren, E. Automatic parcellation of human cortical gyri and sulci using standard anatomical nomenclature. 53, 1–15 (2010).
.. [maier2018] Maier, D., Tekle, K. T., Kifer, M. & Warren, D. S. Datalog: concepts, history, and outlook. in Declarative Logic Programming (eds. Kifer, M. & Liu, Y. A.) 3–100 (Association for Computing Machinery and Morgan & Claypool, 2018). doi:10.1145/3191315.3191317.

