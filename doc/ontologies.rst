.. _ontologies:

Ontologies
==========

Which answers does a query owe when the data are incomplete and an ontology
says what must exist beyond them?  The classical answer, a rewriting of the
query into a union of conjunctive queries, is finite only for restricted
ontology languages and can be exponential even there.  NeuroLang chases the
axioms that create no value as Datalog rules, and rewrites the query with
the others by XRewrite [Gottlob2014]_.

This page states what that procedure computes, why it is split in two, for
which ontologies its answers are complete, and what negation, aggregation
and probabilities mean over the predicates of an ontology.  The approach,
and its use on neuroscience ontologies, is described in [Zanitti2023]_.

.. contents::
   :local:
   :depth: 1

Three instances
---------------

Three ontologies are used throughout this page.

(I) *Deans and units*.  Every dean heads a department, and every
department is a unit:

.. code-block:: turtle

   @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
   @prefix owl: <http://www.w3.org/2002/07/owl#> .
   @prefix : <http://ex.org/> .
   :Dean a owl:Class ; rdfs:subClassOf [
       a owl:Restriction ; owl:onProperty :headOf ; owl:someValuesFrom :Dept
   ] .
   :Dept a owl:Class ; rdfs:subClassOf :Unit .
   :Unit a owl:Class .

with the facts ``Dean('Juan')``, ``headOf('Ana', 'Physics')`` and
``Dept('Physics')``.  Nothing names the department Juan heads.

(II) *A class closed under a property*.  ``A ⊑ ∀p.A``, where ``A`` is a
class and ``p`` a property: whatever an ``A`` reaches through ``p`` is an
``A``.  On the facts ``A(a0)`` and ``p(a0, a1)``, …, ``p(an-1, an)``, with
``a0`` to ``an`` named values, every ``ai`` is an ``A``.

(III) *The Cognitive Atlas*.  The ``neurosynth`` and ``neurostore`` engines
load ``cogat.v2.owl`` (see :doc:`tutorial_cognitive_atlas`).  It gives
10,405 constraints: three for each of its 2,724 ``someValuesFrom``
restrictions, one for each of its 1,797 subclass axioms, 430
``allValuesFrom`` restrictions and 6 ``domain`` and ``range`` axioms.  The 430
``allValuesFrom`` constraints lie in one recursive component of 3,364
predicates.  The engine has no instance of any class.  We add one, of the
class ``CAO_01600``, to measure what happens when data meets the hierarchy.
The restrictions themselves are facts on class names; the 323
``someValuesFrom`` restrictions on ``ro:has_part`` are:

.. nl-repl::

   ans(c, d) :- rdfs:subClassOf(c, r),
       owl:onProperty(r, <http://www.obofoundry.org/ro/ro.owl#has_part>),
       owl:someValuesFrom(r, d)

These instances cover what the procedure must handle: a value no fact names
in (I), a recursion in (II), and the size of a real ontology in (III).  A
fourth, (IV), puts (III) to work on the studies of Neurosynth, in
:ref:`ontologies-open-closed`.

Loading an ontology
-------------------

:meth:`~neurolang.frontend.NeurolangPDL.load_ontology` reads OWL or RDF
files with rdflib and splits them in two.  The *TBox*, the axioms on classes,
becomes constraints, ``body → head`` implications kept apart from the rules
of the program.  The *ABox*, the facts on individuals, becomes relations: the
data added with ``add_tuple_set``, or, with ``individuals=True``, every triple
``s p o`` as the fact ``p(s, o)`` and every ``s rdf:type C`` also as
``C(s)``.  On (I), with the Turtle above in ``deans.ttl`` and loaded from
Python as in :ref:`python-ontology`:

.. code-block:: python

   from neurolang.frontend import NeurolangPDL

   nl = NeurolangPDL()
   nl.load_ontology("deans.ttl")
   nl.add_tuple_set([("Juan",)], name="Dean")
   nl.add_tuple_set([("Ana", "Physics")], name="headOf")
   nl.add_tuple_set([("Physics",)], name="Dept")

:class:`~neurolang.datalog.ontologies_parser.OntologyParser` reads the
axioms of each class ``C`` and each property ``p`` as follows, where ``x``,
``y`` and ``z`` are variables, ``q`` a property, ``D`` and ``B`` classes,
``v`` a value and ``aux`` a fresh predicate:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - OWL axiom
     - constraints
   * - ``C rdfs:subClassOf D``
     - ``C(x) → D(x)``
   * - ``C ⊑ ∃p.D`` (``someValuesFrom``)
     - ``C(x) → ∃y aux(x, y)``, ``aux(x, y) → p(x, y)``,
       ``aux(x, y) → D(y)``
   * - ``C ⊑ ∀p.D`` (``allValuesFrom``)
     - ``C(x) ∧ p(x, y) → D(y)``
   * - ``C ⊑ ∃p.{v}`` (``hasValue``)
     - ``C(x) → p(x, v)``
   * - ``C ⊑ B ⊓ ∃p.D`` (``intersectionOf``)
     - ``C(x) → B(x)`` and the ``someValuesFrom`` constraints
   * - annotation ``C p v``, as ``rdfs:label``
     - ``C(x) → p(x, v)``
   * - ``p rdfs:subPropertyOf q``
     - ``p(x, y) → q(x, y)``
   * - ``p owl:equivalentProperty q``
     - ``p(x, y) → q(x, y)``, ``q(x, y) → p(x, y)``
   * - ``p owl:inverseOf q``
     - ``p(x, y) → q(y, x)``, ``q(y, x) → p(x, y)``
   * - ``p rdfs:domain D``
     - ``p(x, y) → D(x)``
   * - ``p rdfs:range D``
     - ``p(x, y) → D(y)``
   * - ``p a owl:SymmetricProperty``
     - ``p(x, y) → p(y, x)``
   * - ``p a owl:TransitiveProperty``
     - ``p(x, y) ∧ p(y, z) → p(x, z)``

Furthermore, the structure of the TBox is kept as facts on class names,
``neurolang:subClassOf('C', 'D')`` and ``neurolang:label('C', 'v')``, so a
query can walk the hierarchy itself.  ``unionOf``, ``complementOf``,
``equivalentClass``, enumerated classes and cardinality restrictions are
read with a warning and dropped, as is a property axiom on a class
expression, such as the range ``A ⊔ B`` or the anonymous inverse
``[ owl:inverseOf p ]``.  A datatype range, as ``xsd:integer``, gives no
constraint, and property chains, functional and reflexive properties are
not read.  Dropping
an axiom removes answers, never adds one.  Hence the constraints NeuroLang
uses are a subset of the ontology's, and every answer is entailed by the
ontology.

Open world and certain answers
------------------------------

A relation of a Datalog program is closed: what is not derived is false.
An ontology is read under the open-world assumption: the facts are a part of
the world, and the constraints say what else the world must contain.  The
answers a query owes are its *certain answers*, the tuples of named values
true in every world that contains the facts and satisfies the constraints.
A predicate of the ontology's signature of which no tuple is entailed, as
one only in the bodies of constraints and without facts or rules, has the
empty relation as its certain answers.

The *chase* computes them [Fagin2005]_.  It applies each constraint whose
body holds and whose head does not, and stands an unnamed value, a labelled
null, for each ``∃y``.  Its result, the universal model, maps into every
world of the ontology.  Hence a query's certain answers are its answers on
the universal model with no unnamed value [Cali2013]_.

On (I), the chase adds ``aux('Juan', n1)``, ``headOf('Juan', n1)``,
``Dept(n1)`` and ``Unit(n1)``, with ``n1`` unnamed, and ``Unit('Physics')``.
The query

.. code-block:: neurolang

   ans(x) :- headOf(x, y), Unit(y)

has the certain answers ``Ana`` and ``Juan``; ``ans(x, y) :- headOf(x, y),
Unit(y)`` has only ``('Ana', 'Physics')``, as Juan's department has no name.
A constraint ``C ⊑ ∃p.C`` makes the chase infinite, so it is not run as such.
NeuroLang runs the part of it that ends, and rewrites the query for the
rest.

Two mechanisms
--------------

A constraint is *full* if every variable of its head is in its body; it is
*existential* otherwise.  Subclass, ``allValuesFrom``, ``hasValue``,
annotation, property and ``aux(x, y) → p(x, y)`` constraints are full;
``C(x) → ∃y aux(x, y)`` is existential.

**The chase of full constraints.**  A full constraint is a Datalog rule:
``Dept(x) → Unit(x)`` is ``Unit(x) :- Dept(x)``.  Its chase creates no value
and ends, and its result is the least model of the rules.  NeuroLang adds to
the program of a query the full constraints its predicates depend on, among
those whose body can hold from the predicates with facts or rules.  A
predicate with facts that such a rule derives is replaced, in the rules of
the query, by a copy holding its facts and what the rules derive, as Datalog
does not let a rule derive into a relation of facts.

**The rewriting of existential constraints.**  XRewrite [Gottlob2011]_
[Gottlob2014]_ replaces a query atom by the body of a constraint whose head
unifies with it.  An existential constraint applies only if the variable in
its ``∃`` position is a variable of the query that occurs nowhere else and is
not in the head.  A *factorisation* step first unifies atoms that share that
variable.  On (I), the rewriting of the query is the union of eight
rules, among which

.. code-block:: neurolang

   ans(x) :- headOf(x, y), Unit(y)
   ans(x) :- headOf(x, y), Dept(y)
   ans(x) :- aux(x, y), Dept(y)
   ans(x) :- aux(x, y), aux(z, y)
   ans(x) :- aux(x, y)
   ans(x) :- Dean(x)

``Unit(y)`` becomes ``Dept(y)``, then ``aux(z, y)`` by the
full constraint ``aux(z, y) → Dept(y)``; the two ``aux`` atoms are factorised
into ``aux(x, y)``, where ``y`` occurs once; finally, the existential
constraint rewrites ``aux(x, y)`` into ``Dean(x)``.  The rules over
``aux``, which has no values, are dropped before evaluation.

**Dead rewritings.**  Some rewritings are dropped as soon as they appear,
with all those they would give.  Let ``P`` be a predicate without facts or
rules whose constraints are all existential in a position ``i``, as
``aux``, and ``P(…)`` an atom of a rewriting with the term ``t`` in position
``i``.  ``P(…)`` can only be rewritten once ``t`` is a variable that occurs
nowhere else.  A constant, a variable of the head and an occurrence in a
negated atom stay so in every rewriting.  A rewriting step on another atom
keeps ``t``, at the positions of the constraint's variable it unifies with,
since a full constraint has all its head's variables in its body and an
existential one applies to ``t`` in its existential position only if ``t``
occurs once.  Hence ``t`` leaves the other atoms only by becoming the term
in position ``i`` of an atom over ``P`` that is factorised with this one.
NeuroLang follows, for each occurrence of ``t``, the predicates and
positions the rewriting steps that keep it can move it to; if one
occurrence cannot reach position ``i`` of ``P``, every rewriting of the
query has an atom over ``P``, which has no values, and none is evaluated.
On (III) these are the rewritings ``aux_i(s, c), aux_j(z, c)`` of
``measures(s, c), CAO_00587(c)`` with ``i ≠ j``: ``c`` is shared, in the
existential position of both, and neither atom can become the other.  The
rewritings evaluated are the same with and without this step.

The rules of the program that define a predicate in the body of a
constraint are solved with the query, even when the query names only what
the constraint derives: the code a query needs is followed through the
constraints.  With ``Dean(x) :- professor(x), chairs(x)`` in place of the
facts of ``Dean``, ``ans(x) :- headOf(x, y)`` is rewritten into ``ans(x)
:- Dean(x)``, and the rule of ``Dean`` is part of the program that answers
it.

**The split.**  Full constraints are in the rewriting too: (I) needs
``Dept ⊑ Unit`` applied to Juan's unnamed department, which the chase never
creates.  However, they rewrite only an atom with a variable out of the
rule's head, the only kind of atom that can stand for an unnamed value.  An
atom whose variables are all in the head, as ``Dean(x)`` above, stands for
named values only, on which the chase has already applied every full
constraint.  The rewritings left are evaluated, with the chased rules, by
NeuroLang's Datalog engine.  Hence each full constraint reaches the named
values through the chase, and the unnamed ones only through the atoms that
can stand for them.

Two kinds of full constraints depart from this rule.  Firstly, a constraint
whose body is one atom with a variable out of its head, as the domain
``p(x, y) → D(x)``, derives a fact on named values from an atom with an
unnamed one.  On (I), with ``headOf rdfs:domain Manager``,
``Manager('Juan')`` holds through Juan's unnamed department, which the chase
never creates.  NeuroLang rewrites such a constraint as the rule
``Manager(x) :- headOf(x, y)``, whose atom ``headOf(x, y)`` has a variable
out of the head, and adds its rewritings, among which ``Manager(x) :-
Dean(x)``, to the chased rules.  Hence ``Manager('Juan')`` is a named fact
of the chase, and every full constraint applies to it.  Secondly,
transitivity, ``p(x, y) ∧ p(y, z) → p(x, z)``, is applied by the chase
only.  A rewriting step with it replaces ``p(x, y)`` by ``p(x, w) ∧ p(w,
y)``, whose two atoms it rewrites again, without end.

Why split
---------

A union of conjunctive queries is a first-order query.  Hence a finite
rewriting exists only if the certain answers are first-order definable, and
XRewrite ends only when one exists.  Linear and sticky constraints
[Cali2012]_ [Cali2012b]_, among which the DL-Lite family behind OWL 2 QL
[Calvanese2007]_, have that property.  ``allValuesFrom`` does not.  On (II),
``A(ai)`` holds exactly when ``ai`` is reachable from an ``A`` through ``p``.
Reachability is not first-order definable, so no union of conjunctive
queries gives the certain answers of ``ans(x) :- A(x)``; XRewrite produces
``A(x)``, ``A(z) ∧ p(z, x)``, ``A(z) ∧ p(z, w) ∧ p(w, x)``, … without end.
The chase computes it as the transitive closure it is.

Moreover, where the rewriting is finite it can be exponential in the
query: each atom is replaced by any of the classes below its own, and the
alternatives multiply across atoms [Calvanese2007]_.

(III) shows both effects.  The query ``ans(x) :- ro:has_part(x, y)`` is
rewritten with the constraints whose body can hold: none without an
instance, so it runs in 0.7 s.  One instance of ``CAO_01600`` makes 8,566 of
the 10,405 constraints hold.  Rewriting with all of them, as NeuroLang did
before the split, did not end within 120 s.  With the split, the full
constraints run in the chase, the rewriting holds 333 conjunctive queries,
built in 0.8 s, and the query returns the instance in 1.7 s.  rdflib gives
the same answer: a superclass of ``CAO_01600`` has a ``someValuesFrom``
restriction on ``ro:has_part``.

.. list-table:: Time of a query, with one instance of ``CAO_01600``
   :header-rows: 1
   :widths: 70 15 15

   * - query
     - before
     - after
   * - ``ans(x) :- ro:has_part(x, y)``
     - > 120 s
     - 1.7 s
   * - ``ans(x) :- ro:part_of(x, y)``
     - > 120 s
     - 2.3 s
   * - ``ans(x, y) :- ro:part_of(x, y)``
     - 3.0–4.5 s
     - 1.3 s
   * - ``ans(n) :- AGGREGATE[](skos:Concept(c) @ count(c)) = n``
     - > 120 s
     - 1.8 s
   * - ``ans(t) :- cogat_term(c, 'language comprehension'),
       ro:has_part(c, x), cogat_term(x, t)``
     - 13–19 s
     - 2.4 s

The machine ran other work during every run, and times vary by up to a
factor of 2 between runs: "before" gives the range of two runs, "after" the
faster of two, the slower one under 4.3 s for each query.  Without the
instance, the five queries keep their answers and their times, under 2 s.
The chase takes the recursion and the hierarchy; the rewriting is left with
the values the existential constraints create.

When the answers are complete
-----------------------------

The answers are always sound: a chased rule and a rewriting step each derive
only what the ontology entails, and dropping a step only loses answers.
They are the certain answers under three assumptions.

Firstly, the constraints are Horn: one atom in each head, no disjunction.
With ``unionOf`` on the right of a subclass axiom, query answering is
coNP-hard in the data [Calvanese2013]_, and, unless PTime = coNP, no
Datalog program computes it.
NeuroLang drops these axioms, so its answers are then incomplete.

Secondly, no full constraint with two or more atoms in its body derives an
atom over named values only from an atom with an unnamed value.  A
constraint with one such atom is chased with its rewritings, as above.
Under this assumption the named facts of the universal model are those of
the chase of full constraints and of these rewritings.  A query atom that
the rewriting must replace then stands for an atom with an unnamed value,
so it has a variable out of the head: the rewritings left out are those
that only re-derive named facts, and XRewrite's completeness argument
[Gottlob2014]_ goes through with the remaining steps.  An
``intersectionOf``, once read as ``aux(x, y) → B(x)``, is now read as
``C(x) → B(x)``.  The constraints of the parser satisfy the assumption,
with two exceptions.  The first is an
``allValuesFrom`` on a property that leads from an unnamed value to a named
one: a property to which a ``hasValue`` restriction or an annotation gives
a constant, the inverse of a property in a ``someValuesFrom``, or a
symmetric one.  There, an unnamed ``x`` with ``C(x)`` and ``p(x, v)``
makes ``D(v)`` hold for the named ``v``, and NeuroLang does not find it.
On (I), with ``headedBy`` the inverse of ``headOf`` and ``Dept ⊑
∀headedBy.Boss``, Juan is a ``Boss``, and is not returned.  The second is
transitivity, which the rewriting never applies.  With a symmetric and
transitive ``headOf``, ``headOf('Juan', n1)`` and ``headOf(n1, 'Juan')``
give ``headOf('Juan', 'Juan')``, which NeuroLang does not find.  Moreover,
a chain through unnamed values is lost too: with ``A ⊑ ∃p.B``, ``B ⊑
∃p.C`` and a transitive ``p``, ``ans(x) :- p(x, y), C(y)`` misses the
instances of ``A``.  In general, a transitive property makes a certain
answer a question of reachability, which no union of conjunctive queries
expresses, as (II) shows; OWL 2 QL, the profile designed for such
rewritings, has no transitive properties.

Finally, the rewriting ends.  It does on (I) and on (III), with 333
rewritings for ``ro:has_part`` and 615 for ``ro:part_of``.  The rewritings
of the domain and range constraints, computed once for the predicates with
values, are part of the program of every query: if one of them does not
end, no query does.  On random
ontologies of 4 classes whose full constraints are not recursive, the
answers of 1,080 random queries of 1 to 3 atoms match those of a chase with
unnamed values bounded at depth 6; 120 of them are the test
``test_certain_answers_on_random_horn_ontologies``.  With four checks
running at once, three of the 1,080 took between 5 and 11 s, and one more
than 5 minutes, alone: its two atoms over variables out of the head,
``C3(y)`` and ``C2(x)``, are rewritten independently and their
alternatives multiply.  The rewriting need not end at all when recursive
``allValuesFrom`` constraints apply to a variable out of the head, as
``C1 ⊑ ∀p.C2``, ``C2 ⊑ C1`` and ``C1 ⊑ ∃p.C3`` with the query
``ans(y) :- C3(z), p(z, y)``.

The same multiplication sets the time on (III).  With one instance of the
contrast ``cogat:CAO_01210`` (N-back 2-back, which measures some working
memory), ``ans(s) :- cogat:measures(s, c)`` takes 7 s, and ``ans(s) :-
cogat:measures(s, c), cogat:CAO_00587(c)`` 60 s without the dropping of
dead rewritings: the 522 ``someValuesFrom`` restrictions on ``measures``
give as many ``aux_i(s, c)`` for the first atom, the subclasses of working
memory and the restrictions to them as many atoms for the second, and
XRewrite explores their 26,600 pairs.  Dropping the dead ones leaves 224 to
explore, and the query takes 24 s, most of it in building the rewritings
that are then dropped.  A query of three such atoms, ``ans(s) :-
cogpo:has_contrast(s, x), cogat:measures(x, c), cogat:CAO_00487(c)``
(the tasks with a contrast that measures some response inhibition), did
not end within 16 minutes.  In its first 210 s the rewritings kept no more
than three atoms and grew to 2,826, while 146,000 more were built and
dropped as dead: the time goes into building rewritings that are then
dropped.  Testing a rewriting step for this before building its result is
the next step.

Outside these assumptions the answers stay sound.  On 480 more random
queries, of up to 4 atoms, over ontologies with ``hasValue`` and, for half
of them, recursive full constraints, none returned an answer that is not
certain.  One missed an answer, through the ``hasValue`` exception above,
and 7, all over recursive constraints, did not end within 10 s.

The property axioms were checked the same way.  On 720 more random
queries, of 1 to 3 atoms, over the ontologies of the first check with 1 to
3 property axioms added, drawn from all seven kinds for half of the
queries and without transitivity for the others, the bounded chase was run
at depths 3 and 5, and the two agreed.  For 15 queries it did not end
within 30 s; of the 705 others, 699 got the certain answers and none an
answer that is not certain.  Three missed an answer, all through an
``allValuesFrom`` reached back from an unnamed value by a symmetric or
inverse property.  Three did not end within 10 s, all with an
``allValuesFrom`` on a property that an inverse or a domain and a
sub-property axiom also constrain.  The test
``test_certain_answers_with_property_axioms`` runs 48 more.

One more limit concerns rules.  XRewrite rewrites each rule of the program
on its own, so an unnamed value does not pass from one rule to another:
with ``heads(x, y) :- headOf(x, y)``, ``ans(x) :- heads(x, y)`` returns Ana
and not Juan, as ``y`` is in the head of ``heads``.  ``heads(x) :- headOf(x,
y)`` returns both.  A class defined by a rule over a probabilistic
predicate, as ``Dean(x) :- p(x)`` with ``p`` a probabilistic fact, gives no
instance to the constraints of a deterministic query.  These assumptions
delimit the guarantee: complete answers for Horn ontologies in which
``allValuesFrom`` reaches no named value from an unnamed one and no
existential value meets a transitive property, for queries whose rewriting
ends.

Negation and aggregation
------------------------

Negation and aggregation are evaluated on the relations NeuroLang derives:
the chase of the full constraints and the rewritten rules.  They are closed
world over those derived facts, not over the worlds of the ontology.  On
(I), with a relation ``person`` holding Juan, Ana and Eva,

.. code-block:: neurolang

   leads(x) :- headOf(x, y), Unit(y)
   ans(x) :- person(x), ~leads(x)

returns Eva: she is not known to lead a unit, which does not mean the
ontology excludes it.  ``~Unit(y)`` likewise holds of every value not
derived to be a unit.

An aggregate counts the named values of its argument, over the union of the
rewritings of its body.  ``ans(count(x)) :- headOf(x, y)`` gives 2, Ana and
Juan.  ``ans(x, count(y)) :- headOf(x, y)`` gives ``('Ana', 1)`` and no row
for Juan, whose department is unnamed: a count over an ontology predicate
is a lower bound on the count in any world of the ontology.

Ontologies and probabilities
----------------------------

The constraints apply to the deterministic rules of a program, solved
before its probabilistic ones.  A probabilistic rule reads an ontology
predicate through a deterministic rule that queries it.  On (I), with the
probabilistic fact ``funded`` holding Juan with probability 0.5 and Ana with
0.2,

.. code-block:: neurolang

   leads(x) :- headOf(x, y), Unit(y)
   ans(x, PROB(x)) :- funded(x), leads(x)

gives Juan 0.5 and Ana 0.2.  Written directly, ``ans(x, PROB(x)) :-
funded(x), headOf(x, y)`` gives only Ana, because the body of a
probabilistic rule is not rewritten.  The ontology thus acts as certain
knowledge: it defines deterministic relations, and the probabilistic layer
of [Zanitti2023]_ reasons on top of them.

.. _ontologies-open-closed:

Open and closed worlds on the Cognitive Atlas
---------------------------------------------

Which studies used a task that, according to the Cognitive Atlas, measures
response inhibition?  And which form of response inhibition did each one
measure?

The data record particular facts, while an ontology records general ones.
The curators of the Cognitive Atlas state each general fact once and leave
the particulars out.  Read in an open world, these general facts answer
questions about particular studies; read in a closed world, they answer
none.

**Two readings.**  In the *closed world*, what is not recorded is false.  In
the *open world*, what is not recorded is unknown, and the ontology can
imply it.  The Atlas says that every Stroop task has an incongruent −
congruent contrast, that this contrast measures some response inhibition,
and that motor inhibition is a kind of response inhibition.  It names no
study, no contrast of a particular study and no particular instance of a
concept; nor does it decide which form of response inhibition the contrast
measures.  Under the closed world, the curators would have to enter every
study, its contrasts and what each measures before a query found anything.
Under the open world, what they left out stays unknown, and the general
facts still yield answers.

NeuroLang reads the predicates an ontology constrains in the open world.
It returns their *certain answers*, those true in every completion of the
data that agrees with the ontology (section *Open world and certain
answers*).  Each axiom is a *constraint*, an implication ``body → head``
kept apart from the rules of the program (section *Loading an ontology*).
A constraint is *existential* when its head states that some value exists
without naming it; such a value is an *unnamed value*.  The constraints
are applied in two ways (section *Two mechanisms*): the *chase* adds, as
rules would, the facts that the constraints without existentials imply, and
the *rewriting* turns the query into a union of queries over the data that
also finds the answers holding through unnamed values.

(IV) *Stroop studies in the Cognitive Atlas*.  The ``neurosynth`` engine
holds 14,371 studies and loads the Atlas, ``cogat.v2.owl``, as in (III).
The Atlas has no instance of any task.  One rule gives it some: a study
that mentions the Stroop task is an instance of the Atlas class of that
name, ``cogat:CAO_00872`` (Stroop task).

.. code-block:: neurolang

   % cogat:CAO_00872 (Stroop task), cogat:CAO_00487 (response inhibition)
   cogat:CAO_00872(s) :- mentions(s, 'stroop task')

The questions rest on three axioms of the Atlas.  Below, ``A ⊑ B`` reads
"every ``A`` is a ``B``", and ``∃p.D`` reads "related by the property
``p`` to some ``D``":

.. list-table:: The axioms of (IV)
   :header-rows: 1
   :widths: 50 50

   * - In plain English
     - Formally
   * - Every Stroop task has an incongruent − congruent contrast.
     - ``cogat:CAO_00872 ⊑ ∃cogpo:has_contrast.cogat:CAO_01515``
   * - Every incongruent − congruent contrast of the Stroop task measures
       some response inhibition.
     - ``cogat:CAO_01515 ⊑ ∃cogat:measures.cogat:CAO_00487``
   * - Motor inhibition is a kind of response inhibition.
     - ``cogat:CAO_00407 ⊑ cogat:CAO_00487``

In these axioms ``cogat:`` is the namespace
``http://www.cognitiveatlas.org/ontology/cogat.owl#``, and ``cogpo:`` that
of the Cognitive Paradigm Ontology, from which the Atlas takes
``has_contrast``; ``cogat:CAO_00872`` is the Stroop task;
``cogat:CAO_01515`` is its contrast
``Stroop_task-incongruent_minus_congruent``; ``cogat:CAO_00487`` is
response inhibition; finally, ``cogat:CAO_00407`` is motor inhibition.
Every identifier, axiom and count on the Atlas in this section was checked
with rdflib on the engine's copy of the file.  The Stroop task has three
more contrasts, none of which measures anything.  Response inhibition has
one more named subclass, emotional suppression (``cogat:CAO_00283``).  No
contrast of the Atlas measures motor inhibition or emotional suppression,
so the third axiom does not change the answers below.  The file holds no
fact of ``cogpo:has_contrast`` or ``cogat:measures`` between individuals,
and no individual of ``cogat:CAO_00487`` or ``cogat:CAO_00872``.

The closed world is computed on the same engine with the constraints
removed and ``cogpo:has_contrast``, ``cogat:measures`` and
``cogat:CAO_00487`` as empty relations: the plain Datalog reading of the
same rules.  The results were computed on 14 November 2026 with NeuroLang
master ``37acbf99`` merged with #1250 (``1d49ac93``), one query at a time;
each time includes the compilation of the query.

**Which studies measure response inhibition?**  The question, as a query:

.. code-block:: neurolang

   % cogat:CAO_00872 (Stroop task), cogat:CAO_00487 (response inhibition)
   cogat:CAO_00872(s) :- mentions(s, 'stroop task')
   inhibition_study(s) :- cogpo:has_contrast(s, x), cogat:measures(x, c), cogat:CAO_00487(c)
   ans(s) :- inhibition_study(s)

and its first two steps, ``ans(s) :- cogpo:has_contrast(s, x)`` and
``ans(s) :- cogpo:has_contrast(s, x), cogat:CAO_01515(x)``, each after the
same rule for ``cogat:CAO_00872``:

.. list-table:: Studies with a contrast that measures response inhibition
   :header-rows: 1
   :widths: 34 22 22 22

   * - Query
     - Closed world
     - Open world
     - Why
   * - ``has_contrast(s, x)``
     - 0 (no rows), 0.3 s
     - 139 rows, 3.4 s
     - every Stroop task has some contrast
   * - ``has_contrast(s, x), CAO_01515(x)``
     - not run
     - 139 rows, 15.1 s
     - that contrast is an incongruent − congruent one
   * - ``inhibition_study(s)``
     - 0 (no rows), 0.5 s
     - did not end within 2,400 s
     - the rewriting of three atoms does not end in time

.. list-table:: Open world, ``has_contrast(s, x)``: first rows (139 rows)
   :header-rows: 1
   :widths: 15 85

   * - ``s``
     - title (``study_title``)
   * - 11532889
     - Anterior cingulate cortex and response conflict: effects of response
       modality and processing domain.
   * - 11844731
     - Imaging how attention modulates pain in humans using functional MRI.
   * - 11919006
     - An event-related functional MRI study comparing interference effects
       in the Simon and Stroop tasks.
   * - 12595201
     - Practice-related effects demonstrate complementary roles of anterior
       cingulate and prefrontal cortices in attentional control.
   * - 12738340
     - Brain activation patterns during a selective attention test-a
       functional MRI study in healthy volunteers and patients with
       schizophrenia.
   * - …
     - 139 rows

The first two steps find, through an unnamed contrast, the 139 studies the
closed world cannot.  The curators stated one axiom; they did not enter 139
contrasts.  By the two axioms, the same 139 studies are the certain answers
of ``inhibition_study(s)``: the contrast of each measures some response
inhibition.  This follows from the axioms, checked with rdflib; the engine
did not compute it.  The rewriting of the three-atom query explores more
rewritings than it can within the time, as the section *When the answers
are complete* describes for the same query with one instance of a
contrast.

**Which form of response inhibition did each one measure?**  In the open
world:

.. code-block:: neurolang

   % cogat:CAO_00872 (Stroop task), cogat:CAO_00487 (response inhibition)
   cogat:CAO_00872(s) :- mentions(s, 'stroop task')
   ans(s, c) :- cogpo:has_contrast(s, x), cogat:measures(x, c), cogat:CAO_00487(c)

In the closed world, at the level of classes: the restrictions of the
Atlas are facts on class names (``rdfs:subClassOf``, ``owl:onProperty``,
``owl:someValuesFrom``), which the engine's ``cogat_measures``,
``cogat_subclass_of`` and ``cogat_term`` already walk:

.. code-block:: neurolang

   has_contrast_class(t, k) :- rdfs:subClassOf(t, r), owl:onProperty(r, <http://www.cogpo.org/ontologies/working/CogPOver2011.owl#has_contrast>), owl:someValuesFrom(r, k)
   inhibition(c) :- cogat_term(c, 'response inhibition')
   inhibition(c) :- cogat_subclass_of(c, d), cogat_term(d, 'response inhibition')
   inhibition_task(t, c) :- has_contrast_class(t, k), cogat_measures(k, c), inhibition(c)
   ans(t, n, m) :- inhibition_task(t, c), cogat_term(t, n), cogat_term(c, m)

and, with the same first four rules, the studies that mention such a task,
``ans(s, n) :- inhibition_task(t, c), cogat_term(t, n), mentions(s, n)``.

.. list-table:: The form of response inhibition each study measured
   :header-rows: 1
   :widths: 30 20 50

   * - Query
     - Rows, time
     - Why
   * - open world, ``ans(s, c)``
     - 0 (no rows), 181.7 s
     - each study's response inhibition is an unnamed value
   * - closed world, tasks ``ans(t, n, m)``
     - 7 rows, 5.8 s
     - the class axioms, read as facts, name the concept
   * - closed world, studies ``ans(s, n)``
     - 139 rows, 6.4 s
     - of the 7 task names, only ``'stroop task'`` is a Neurosynth term

.. list-table:: Closed world, ``ans(t, n, m)`` (7 rows)
   :header-rows: 1

   * - ``t``
     - ``n``
     - ``m``
   * - ``cogat:CAO_00707``
     - color-word stroop task
     - response inhibition
   * - ``cogat:CAO_00794``
     - conditional stop signal task
     - response inhibition
   * - ``cogat:CAO_00810``
     - choice reaction time task
     - response inhibition
   * - ``cogat:CAO_00827``
     - go-no-go task
     - response inhibition
   * - ``cogat:CAO_00864``
     - simon task
     - response inhibition
   * - ``cogat:CAO_00872``
     - stroop task
     - response inhibition
   * - ``cogat:CAO_00950``
     - hayling sentence completion test
     - response inhibition

The open world adds answers, never values.  It can say that a study
measures some response inhibition, but it never invents which one: the
Atlas names none for any study.  The closed world names the concept because
the class axioms, read as facts, name it, and it reaches studies only
through a second, textual step, the term of the task's name.

**Why not** ``mentions(s, 'stroop task')``?  On Neurosynth it returns the
same 139 studies in 2.2 s.  The question, however, is about a concept, and
the ontology decides which tasks count: a curator who gives another task a
contrast that measures response inhibition changes the answer without
changing the query.  Of the 7 task names above, only ``'stroop task'`` is
in the Neurosynth vocabulary; hence on Neurosynth the example shows the
mechanism, not a gain in recall.  The vocabulary of the ``neurostore``
engine has five of them: its TF-IDF table puts ``'stroop task'`` above the
threshold of ``mentions`` in 444 studies, and the five names in 742.  The
open-world query was not run there, as it does not end on Neurosynth with
one task.

**Where in the brain?**  The open-world predicate as the condition of a
probabilistic query:

.. code-block:: neurolang

   % cogat:CAO_00872 (Stroop task), cogat:CAO_00487 (response inhibition)
   cogat:CAO_00872(s) :- mentions(s, 'stroop task')
   inhibition_study(s) :- cogpo:has_contrast(s, x), cogat:measures(x, c), cogat:CAO_00487(c)
   p_active_given_inhibition(i, j, k, PROB) :- (activated(i, j, k, s) & selected_study(s)) // (inhibition_study(s) & selected_study(s))
   ans(i, j, k, p) :- p_active_given_inhibition(i, j, k, p)

Here ``p`` is the probability that a study, among those that measure
response inhibition, reports a peak within 6 mm of the voxel
``(i, j, k)``.

.. list-table:: Activation given response inhibition
   :header-rows: 1
   :widths: 30 25 45

   * - Reading
     - Rows, time
     - Why
   * - closed world
     - 0 (no rows), 5.1 s
     - the condition holds of no study
   * - open world
     - did not end within 2,400 s
     - its condition is the three-atom query above

The certain answers of the condition are the 139 Stroop studies.  Hence
the same conditional over ``mentions(s, 'stroop task')``,
``p_active_given_stroop(i, j, k, PROB) :- (activated(i, j, k, s) &
selected_study(s)) // (mentions(s, 'stroop task') & selected_study(s))``,
gives what the open-world query owes: 1,296,985 voxels with ``p > 0``, in
14.2 s.  Its highest ``p`` in each region of the Destrieux atlas, with the
MNI coordinates of the voxel that reaches it, are:

.. list-table:: ``p_active_given_stroop``, the six Destrieux regions with the highest ``p``
   :header-rows: 1

   * - Destrieux region
     - MNI (mm)
     - ``p``
   * - L S precentral inf part
     - (−43, 5, 31)
     - 0.180
   * - L G front sup
     - (−4, 11, 49)
     - 0.173
   * - L S front inf
     - (−42, 20, 28)
     - 0.151
   * - R G front sup
     - (3, 19, 43)
     - 0.151
   * - R G and S cingul Mid Ant
     - (5, 22, 37)
     - 0.151
   * - L G and S cingul Mid Ant
     - (−6, 12, 48)
     - 0.137

The highest, 0.180, is 25 of the 139 studies.

**What the curators did not state.**  The stop signal task,
``cogat:CAO_00871``, has six contrasts in the Atlas, and none has a
``measures`` axiom.  ``'stop signal'`` is a Neurosynth term:

.. code-block:: neurolang

   % cogat:CAO_00871 (stop signal task), cogat:CAO_00487 (response inhibition)
   cogat:CAO_00871(s) :- mentions(s, 'stop signal')
   inhibition_study(s) :- cogpo:has_contrast(s, x), cogat:measures(x, c), cogat:CAO_00487(c)
   ans(s) :- inhibition_study(s)

.. list-table:: Stop signal studies
   :header-rows: 1
   :widths: 34 22 44

   * - Query
     - Rows, time
     - Why
   * - closed world, ``mentions(s, 'stop signal')``
     - 99 rows, 2.3 s
     - the term is in their abstracts
   * - open world, ``inhibition_study(s)``
     - 0 (no rows), 1.8 s
     - no stop signal contrast measures anything in the Atlas

Non-exhaustiveness is allowed; missing knowledge is not filled in.  The
open world completes the instances of what the curators stated, and it
cannot infer what they never stated, that a stop signal contrast measures
response inhibition.  The answers are as complete as the ontology.

Writing such queries
~~~~~~~~~~~~~~~~~~~~

Keep a chain through unnamed values inside one rule.  The rewriting treats
each rule on its own (section *When the answers are complete*), so
``inhibition_study(s)`` holds the whole chain and projects it onto ``s``; a
rule with ``x`` or ``c`` in its head loses the answers reached through the
unnamed contrast.  Define the instances of a class with a deterministic
rule, as with ``mentions`` above, and bring ``PROB`` in afterwards: a class
defined by a rule over a probabilistic predicate gives no instance to the
constraints of a deterministic query (same section).

In short, the open world lets the curators of an ontology be
non-exhaustive.  On (IV), one axiom stated once gave 139 studies a contrast
that no one entered, and only what was never stated stayed out of reach.
The three-atom chain is entailed but not yet computed: a rewriting that
ends on it, as the section *When the answers are complete* calls for, would
confirm it.

What would confirm and what limits this
---------------------------------------

The completeness claim rests on an argument and on 2,265 random queries,
not on a mechanised proof.  The W3C OWL test cases [OWLTest2004]_ check
the reading of OWL itself: of the 84 tests of the approved and proposed
archives, the 8 entailment and non-entailment tests whose premises are in
the fragment NeuroLang reads and whose conclusion is a set of facts on
individuals (on ``allValuesFrom``, ``someValuesFrom``, ``inverseOf``,
``SymmetricProperty``, ``TransitiveProperty`` and an unread
``owl:imports``) pass, as ``test_w3c_owl_tests.py``.  The 76 others are
skipped, each with its reason in
``frontend/tests/w3c_owl_tests/skipped.tsv``: 13 consistency and
inconsistency tests (NeuroLang does not check consistency), 7 on
``owl:imports``, 5 not about an OWL feature, 44 whose premises use what
NeuroLang does not read (functional and inverse functional properties,
``owl:samePropertyAs``, cardinalities, equality of individuals and
classes, ``unionOf``, ``complementOf``, ``oneOf``, ``disjointWith``,
``intersectionOf`` other than as a superclass, a class expression as a
subclass) and 7 whose conclusion is an axiom on classes or properties
rather than a query.  The OWL 2 test repository no longer
resolves.  A cross-check against an OWL reasoner, such as
HermiT or Konclude, on ontologies with instances of the Cognitive Atlas,
would test it at scale.  The termination of the rewriting on recursive
``allValuesFrom`` is not guaranteed.  A Datalog rewriting of the anonymous
part, as in the combined approach [Lutz2009]_ [Kontchakov2011]_, which
materialises one unnamed value per existential constraint and filters the
spurious answers, would remove that limit, at the cost of a filtration step
for each query.  Until then, the guarantee holds for Horn ontologies in
which ``allValuesFrom`` reaches no named value from an unnamed one and no
existential value meets a transitive property, on queries whose rewriting
ends.

.. rubric:: References

.. [Cali2012] A. Calì, G. Gottlob and T. Lukasiewicz. *A general
   Datalog-based framework for tractable query answering over ontologies*.
   Journal of Web Semantics, 14, 57–83, 2012.
   doi:10.1016/j.websem.2012.03.001

.. [Cali2012b] A. Calì, G. Gottlob and A. Pieris. *Towards more expressive
   ontology languages: The query answering problem*. Artificial
   Intelligence, 193, 87–128, 2012. doi:10.1016/j.artint.2012.08.002

.. [Cali2013] A. Calì, G. Gottlob and M. Kifer. *Taming the infinite
   chase: Query answering under expressive relational constraints*.
   Journal of Artificial Intelligence Research, 48, 115–174, 2013.
   doi:10.1613/jair.3873

.. [Calvanese2007] D. Calvanese, G. De Giacomo, D. Lembo, M. Lenzerini and
   R. Rosati. *Tractable reasoning and efficient query answering in
   description logics: The DL-Lite family*. Journal of Automated Reasoning,
   39(3), 385–429, 2007. doi:10.1007/s10817-007-9078-x

.. [Calvanese2013] D. Calvanese, G. De Giacomo, D. Lembo, M. Lenzerini and
   R. Rosati. *Data complexity of query answering in description logics*.
   Artificial Intelligence, 195, 335–360, 2013.
   doi:10.1016/j.artint.2012.10.003

.. [Fagin2005] R. Fagin, P. G. Kolaitis, R. J. Miller and L. Popa. *Data
   exchange: semantics and query answering*. Theoretical Computer Science,
   336(1), 89–124, 2005. doi:10.1016/j.tcs.2004.10.033

.. [Gottlob2011] G. Gottlob, G. Orsi and A. Pieris. *Ontological queries:
   Rewriting and optimization*. In Proceedings of the 27th IEEE
   International Conference on Data Engineering (ICDE), 2–13, 2011.
   doi:10.1109/ICDE.2011.5767965

.. [Gottlob2014] G. Gottlob, G. Orsi and A. Pieris. *Query rewriting and
   optimization for ontological databases*. ACM Transactions on Database
   Systems, 39(3), 1–46, 2014. doi:10.1145/2638546

.. [Kontchakov2011] R. Kontchakov, C. Lutz, D. Toman, F. Wolter and
   M. Zakharyaschev. *The combined approach to ontology-based data access*.
   In Proceedings of the 22nd International Joint Conference on Artificial
   Intelligence (IJCAI), 2656–2660, 2011.
   doi:10.5591/978-1-57735-516-8/IJCAI11-442

.. [OWLTest2004] J. J. Carroll and J. De Roo (eds.). *OWL Web Ontology
   Language Test Cases*. W3C Recommendation, 10 February 2004.
   https://www.w3.org/TR/owl-test/

.. [Lutz2009] C. Lutz, D. Toman and F. Wolter. *Conjunctive query answering
   in the description logic EL using a relational database system*. In
   Proceedings of the 21st International Joint Conference on Artificial
   Intelligence (IJCAI), 2070–2074, 2009.

[Zanitti2023]_ is listed in :doc:`tutorial_squall`.
