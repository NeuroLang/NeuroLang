
Get Started with NeuroLang
==========================


First Steps With NeuroLang
--------------------------
NeuroLang is a unifying formalism to perform complex queries and explorations using heterogeneous data sources like tabular data,
volumetric images, and ontologies. To perform this in sound manner, NeuroLang is a probabilistic logic programming language based on Datalog [abiteboul1995]_, [maier2018]_. A simple example can be produced by analysing the NeuroSynth database, to find all article titles which have the word "pain" in the abstract. For this, we start by importing the NeuroLang deterministic solver::

    >>> from neurolang.frontend import NeurolangDL
    >>> 



.. [abiteboul1995] Abiteboul, S., Hull, R. & Vianu, V. Foundations of databases. (Addison Wesley, 1995).
.. [maier2018] Maier, D., Tekle, K. T., Kifer, M. & Warren, D. S. Datalog: concepts, history, and outlook. in Declarative Logic Programming (eds. Kifer, M. & Liu, Y. A.) 3–100 (Association for Computing Machinery and Morgan & Claypool, 2018). doi:10.1145/3191315.3191317.

