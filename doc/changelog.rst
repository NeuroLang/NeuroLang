.. _changelog:

Changelog
=========

All notable changes to NeuroLang are documented here.

NeuroLang is in alpha.  Pre-release versions (``0.1aN``) are published on
`PyPI <https://pypi.org/project/neurolang/#history>`_; the *Unreleased*
section below lists the changes on the current ``master`` branch that are
not yet part of a numbered release.

Contributors: when adding a change, add an entry under *Unreleased* using the
format below, then move it to the appropriate version section when releasing.


Unreleased
----------

**New features:**

* Ontologies: the property axioms ``subPropertyOf``, ``equivalentProperty``,
  ``inverseOf``, ``domain``, ``range``, ``SymmetricProperty`` and
  ``TransitiveProperty`` are read as constraints without existential
  variables, applied by the chase.  A domain or range constraint, whose body
  has a variable out of its head, is chased with its rewritings, so
  ``Manager('Juan')`` holds when Juan heads an unnamed department of a
  ``headOf`` whose domain is ``Manager``.  Transitivity is never a rewriting
  step; its answers through unnamed values are outside the completeness
  guarantee of :doc:`ontologies`.  A predicate only in the bodies of
  constraints, with no facts or rules, answers the empty relation.
  
* The query assistant names the columns of its answers after what they
  hold, ``ans(region, probability)`` rather than ``ans(l, p)``, and gives
  the entities an answer is computed over that the question names (the
  regions at the ends of a tract, the parcels of a network) as answers of
  their own, named queries ``ans[name](...)``: its syntax reference and
  ReAct steps say so, and the examples of the banks are rewritten so
  (a test checks their column names).  New example on ``neurosynth_iit``:
  the Bayes factor of the cognitive tasks mentioned in the studies
  activating the regions at both ends of the left uncinate fasciculus,
  with these regions.

* Ontologies: the constraints without existential variables (subclass,
  ``allValuesFrom``, ``hasValue``, annotations) are applied by the chase, as
  Datalog rules; XRewrite rewrites the query with the existential ones, the
  others only on atoms that can stand for a value they introduce.  A query
  on the Cognitive Atlas with an instance of a class high in its hierarchy
  now ends (``ans(x) :- ro:has_part(x, y)``: 1.7 s, was > 120 s).  New page
  :doc:`ontologies`.  Fixes on the way: XRewrite's factorisation dropped the
  rest of the body, a rewriting step bound a query variable only in the
  rewritten atom (wrong answers with ``hasValue``), a renamed constraint
  variable could take a query variable's name, a negated atom made XRewrite
  fail, the chased predicates broke ``solve_all``, an aggregate over a
  rewritten body counted per rewriting, and ``C ⊑ B ⊓ ∃p.D`` is read as
  ``C(x) → B(x)``.
* ``neurolang-repl-serve --local`` loads the text embeddings of an engine
  when it builds it (its documents', from their cache or computed into it,
  and the SPECTER2 query model): with ``--users`` in the builder, before
  forking the sessions, which share them; their first ``query_similarity``
  (``similar_study``, ``lift_llm``) no longer loads the model, and no
  longer crashes a session on macOS. ``--no-preload-embeddings``
  (``preload_embeddings: false``) leaves them to the first query, as
  before. Queries are embedded with the numpy SPECTER2 encoder whenever its
  weights are next to the model, not only without torch; encoders of an
  engine built again share torch's model.
* The network of the brain viewer names what is under the mouse on a
  slice: a node, the name of its region; an edge, the names of the two
  regions it joins and its value (within ``hover_mm`` of ``viewer.yaml``,
  2 mm, of the sphere or tube drawn).
* ``neurolang-repl-serve --activity-db`` gives the admins an activity page,
  ``/admin/activity``: filters of the period, user and engine above
  figures (uptime, users, inputs and failures, CPU time, LLM tokens, the most
  memory of a session), a strip of the runs of the server, charts of the
  inputs, CPU, tokens and active users over time (Chart.js, served by the
  server), the users, and the inputs, sortable, searchable and each opening
  on its result. Datasette (``--dashboard``) stays, as *Raw data and SQL*.
  Its queries of the last days no longer take the whole first day (they
  compared ISO times with SQLite's), and its uptime counts from the first
  run logged.
* The engines ``neurostore_tracts`` and ``neurostore_iit``: NeuroStore with
  tractography, as ``neurosynth_tracts`` and ``neurosynth_iit`` are
  Neurosynth with it (the vocabulary of tracts; the IIT whole-brain
  tractogram and tract_querier's bundles). The page of the REPL offers
  ``neurostore_iit`` as *NeuroStore + tracts*, with the text embeddings of
  ``neurostore``. What they share is two abstract engines,
  ``meta_analysis_tracts`` and ``iit_tractogram``.
  
* The login page of ``neurolang-repl-serve --users`` is the REPL's: its logo
  and colours, in the theme the visitor chose in the REPL (or the system's), a
  card with the name and the password, which can be shown, a button that waits
  while the server answers, the error above the form, *Sign in with ORCID* and
  *Continue as a guest* under *or*, and the request for an account as a card of
  its own. It is still one file, with the logo in it: the server refuses every
  other file before a login.

* ``neurolang-repl-serve --local`` prepares each engine before serving
  (``--no-prepare`` not to): it builds it once, in a process of its own, its
  downloads and its tables kept in ``--data-dir``, so that a session starts
  from them without the network; one it cannot prepare is told, and the
  server serves the others.  The bundle of ``neurolang-repl-serve`` keeps what
  its engines download in ``--downloads`` (``~/.cache/neurolang/
  repl-downloads``), so that a rebuild does not need its sources (OSF,
  GitHub) to be up.

* ``neurolang-repl-serve --config server.yaml``: the server's options in a
  YAML file, each by its name, its paths relative to it, the command line
  replacing them; the service installed with it runs the file.  Its secrets
  (the ORCID client, the server's ``env``, the ``password``, the
  ``key_password`` of the certificate's key) in a second YAML file,
  ``secrets: secrets.yaml`` or ``--secrets``, which the server refuses if
  others may read it.
* ``neurolang-repl-serve --users users.yaml --orcid orcid.yaml``: the users
  named by an ORCID iD sign in with ORCID (OAuth 2.0, scope
  ``/authenticate``), with no password, their ORCID name shown; an iD of no
  user asks for an account, with a message, which an admin approves in the
  console (*Account requests*, adding them to groups) or with
  ``--approve-request ID --groups G``, or declines.
* The admin console of ``neurolang-repl-serve`` merges an ORCID user into a
  user of a name of their own (*merge into…*, ``--merge-orcid ID USER``),
  who then logs in with their password or with ORCID (their ``orcid``), in
  the groups of both, and separates them again (*Separate ORCID*,
  ``--separate-orcid USER``): the iD a user of its own, in the groups but
  ``admin``.
* On their account page, a user of ``neurolang-repl-serve`` links their
  ORCID iD (*Link my ORCID iD*, signing in at ORCID; the account of the iD,
  if any, merged into theirs) and unlinks it; a user who signs in with
  ORCID only merges their account into one of theirs with a password,
  giving its name and password.
* ``neurolang-repl-serve --local --users`` tracks the CPU seconds and the
  memory of each session's process, its own apart from the engine's data it
  shares with the builder (on Linux), shown in the admin console; and, with
  ``--activity-db FILE``, logs what each user runs, its result as text, how
  long it took and the resources it took, in SQLite, with the totals of each
  user and their log in the console. The page tells its users their inputs
  are logged.
* ``neurolang-repl-serve --local --users`` runs at most ``--max-running``
  queries at once (the machine's CPUs), of all the sessions, the others
  waiting their turn, their page saying so; and stops a query running longer
  than ``--query-minutes`` (60), ending its session's process.
* The REPL page of ``neurolang-repl-serve --local --users`` brings back a
  session the server ended (unused, made room for another, its query ran too
  long) at its next input: it starts it again, runs
  again the rules of the session, as after *Stop*, then that input. The
  server answers an ended session with 409, ``{"ended": true}``.
* ``neurolang-repl-serve --dashboard`` serves the admins a dashboard of
  ``--activity-db`` and the ratings at ``/admin/logs/``: Datasette, with
  datasette-vega's charts and queries of the activity by hour, user and
  engine, the slowest and failed inputs, the ends of the sessions and the
  uptime (the ``dashboard`` extra). ``GET /health`` tells a monitor whether
  the server serves, without a login; ``--daemon status`` what the service
  manager says of it; and the activity log has each run of the server, to
  its stop (a SIGTERM, as from systemd, now stops the server as Ctrl+C
  does, ending its sessions).
* With ``--activity-db``, an answer of the assistant logs the tokens it sent
  and received, its provider and model, and whose key paid (the page's, or
  the server's), shown by user in the console and by user, model and key
  in the dashboard. The REPL's reply carries them (``tokens``).
* The REPL in the browser is redesigned: the question and the query on the
  left, the answer on the right (one under the other on a narrow screen,
  which scrolls to the answer), in the colours of the logo and of the
  documentation, light and dark. A first visit finds an example in the box;
  the result pane says what comes there until a query runs; probabilities are
  drawn as tracks filled to their value, the most probable in red; the
  session commands are in a *More* menu.
* The tables of the browser REPL have a *Plot* button: bars, a histogram,
  points or a line of their columns, of the rows the filter keeps.
* A long rule the browser REPL puts in its box (an example, the assistant's
  query, a follow-up) is laid out for the box's width: its head on the left
  and an atom of its body a line, after ``:-``; a long probability breaks
  before its ``//``. *More* › *Format the query* (Alt+Shift+F) does it to
  what is typed.
* The graph of a query (``.graph``, the *Graph* tab) opens fitted to the
  page, its whole in view, and zooms (−, *Fit*, +, *Actual size*, or Ctrl
  or Cmd and the wheel). A relation of an ontology is drawn as a node of a
  knowledge graph, linked to others; the name of a table is below the rim
  of its cylinder.
* ``.view`` of pairs of regions or points with several value columns
  (``ans(a, b, p, q)``) draws the network of the first, the probability
  first, rather than every pair at 1; the viewer's *Edge value* chooses the
  column the edges are of, its edge threshold following.
* The brain viewer shows a range of values: a low and a high threshold
  (values or percentiles), with two sliders over a histogram of the
  values, the range shown in the colours it is drawn with.
* In the browser REPL, a result has a *Brain* tab beside *Table* and
  *Graph*: it shows the result in the brain viewer.
* The assistant takes at most 16 steps by default (``max_steps``, was 8),
  for questions that need more tool calls before a query that runs.  The
  defaults of the LLM configuration are in one file,
  ``utils/llm/defaults.yaml``, which the configuration reads; the example
  configuration and the docs quote it, and the tests check that they agree.
  So are the other tunable defaults: those of the brain viewer and
  ``.view`` (``utils/tui/viewer.yaml``), the options of
  ``neurolang-repl-serve`` (``tools/serve_repl/defaults.yaml``) and the
  limits of the browser REPL's page (``doc/_static/repl/settings.yaml``,
  which the page reads at start).
* When the assistant of the browser REPL finds no query, a notice under the
  question says so, why (the steps it used up, its last query's error, or
  in its words what it found), that the query box still holds the query of
  before, and what to try: edit the question, edit its last query, see its
  steps, browse the examples.
* The browser REPL keeps the questions asked in English, each with the
  query the assistant gave, between visits (*Past questions*, under the
  box of the assistant): *Use this query* puts it back in the box, in its
  language, with what the assistant said of it and the question it answers;
  *Ask again* puts the question back.
* The query assistant answers a question naming a cognitive process, task
  or concept ("studies about working memory", "the n-back task") without
  saying mention, term or word with the studies closest to the concept's
  Cognitive Atlas definition (``cogat_definition`` with
  ``similar_study``), looked up with ``search_values`` over
  ``cogat_term``, or to the question's own words when the atlas does not
  have it; ``mentions`` is kept for the questions about mentions, terms
  or words.  The guidance and the examples of the
  ``coordinate_meta_analysis`` engines say so.

**Performance:**

* The query assistant caches its prompts on the Anthropic API, and on
  OpenRouter for Claude: the steps of
  an answer repeat the system prompt and the conversation so far, which are
  then read from the cache at a tenth of the price of input tokens or less,
  making an answer about three times cheaper.  The line after the query
  counts the tokens read from the cache, also for the providers that cache
  by themselves, such as DeepSeek on OpenRouter.

* A natural join on a unique key of strings, such as the NeuroStore study
  ids joined with the probabilities of ``selected_study``, looks the keys
  up by their codes: ``uniformity('pain', i, j, k, u)`` on ``neurostore``
  takes 23 s instead of 81 s (2 mm grid).
* A function of a rule's variables, such as ``query_similarity(q, s)``
  over the 65,000 studies of NeuroStore, is given each row as a tuple read
  by column, not as a pandas row: ``uniformity_llm('auditory stimuli', i,
  j, k, u)`` on ``neurostore`` spends 1 s instead of 15 s on it.

**Bug fixes:**

* The semi-naive chase missed tuples in the first iteration of a recursive
  stratum: a rule's recursive atom saw only the tuples found earlier in that
  iteration, not those of earlier strata.  With ``rr(y, x) :- rr(x, y)``,
  ``p(x, y) :- rr(x, y)`` and ``rr(x, y) :- p(x, y)``, ``p`` lacked the
  first tuple of ``rr``.  In the first iteration a recursive atom now sees
  its whole relation.
* An engine extending another with the same ``text_embeddings`` entry
  (``neurosynth_iit``, ``neurostore_iit``, the ``_tracts`` ones) shares its
  caches (document embeddings, PubMed texts, model copies) in the
  ``embeddings`` directory of the most basic such engine
  (``engine_registry.embeddings_owner``), instead of embedding all the
  abstracts again under its own name, which took hours on NeuroStore.  An
  engine whose documents differ from the shared embeddings, or whose own
  directory alone has them, keeps its own.
* Arithmetic in the head of a rule (``ans(x, n / 2) :- r(x, n)``) was
  taken for an aggregation, the operator applied to each group's values,
  and gave rows of missing values or failed in pandas; it is now refused
  with a ``ForbiddenExpressionError`` that tells how to compute the value
  in the body, ``(v == n / 2)``.

* The threshold sliders of the brain viewer sit over the values they set
  on the histogram. In percentile mode a thumb was placed at its percentile
  along the slider while the histogram's axis is the value, so for skewed
  maps the low thumb could be far from its line (128 px of 360 at the 50th
  percentile of cubed values); the thumbs are now placed by value, and the
  histogram is inset by half a thumb so its ends meet the thumbs' travel.
* Checking or running a query on an ontology's predicates, as
  ``ans(t) :- cogat_term(c, 'working memory'), ro:has_part(c, x),
  cogat_term(x, t)`` on the neurosynth engine, takes seconds: stratifying
  the ontology's 6.5k rules, one recursive component of 3364 predicates,
  ran for more than 30 minutes.
* ``.view`` of streamlines with the regions they join, as
  ``ans(s, l1, l2) :- connects(s, l1, l2)``, draws the streamlines, their
  column of the sort ``streamline``: it drew the network of the regions.
  The engine ``tracts`` declares the relations of a tractogram
  (``endpoint_point``, ``streamline_label``...), so that their streamline
  is of that sort.
* A disjunction of functions whose variables a positive atom outside it
  binds, as in ``ans(l) :- tian(l, r), (startswith('aHIP', l) |
  startswith('pHIP', l))``, runs: it failed with ``UnboundVariablesError``.
* A probabilistic rule over another whose head repeats a variable, as
  ``b(c, s) :- a(c, s), roi(s)`` with ``a(s, s) :- term(s),
  selected_study(s)``, keeps all its body: the rule's variables were not
  renamed apart from the query's when its head was unified, and ``roi(s)``
  was dropped (the probability, and the replicates of ``RESAMPLE``, were
  those of ``term`` alone).

**Bug fixes:**

* The query assistant's guidance on the meta-analysis engines: the regions
  not activated by the studies about a topic get their probability of not
  being activated given the topic (``~destrieux_activated`` over
  ``selected_study``), not the regions no such study activates; and MNI
  coordinates are never the voxel indices of ``activated``, ``lift`` or
  ``brain_mask``.

**Bug fixes:**

* The query assistant (ReAct) stops search loops: after two
  ``search_values`` calls finding no value it had not seen (a repeated
  call, a variant of a pattern), the observation asks for a query with
  the values found, run with ``run_query``.

**Bug fixes:**

* The memory limit of a query the assistant runs (``query_memory_mb``,
  4 GB) counts the memory private to the forked process running it, not
  the engine's relations it reads, which it shares with its parent: on
  Linux the resident memory of the child is the parent's from the fork, so
  every query of an engine larger than the limit was stopped as using too
  much memory; on macOS a query reading ``activated`` (346 M rows on the
  1 mm grid with the 6 mm kernel) was.

**Bug fixes:**

* The assistant's ``search_values`` searches a predicate whose columns have
  the same name, as ``cogat_subclass_of(iri, iri)``: it failed with
  ``'DataFrame' object has no attribute 'str'``.

**Performance:**

* A Python function computing a column of a rule, as ``query_similarity``,
  runs once per distinct row (pandas and polars backends): it ran on the
  duplicates of its input, 2 M rows instead of 14 k in
  ``cogat_definition('memory', d), similar_study(d, s),
  study_journal(s, 'Neuron')``, which took 70 s instead of 3 s.


v0.1a22 (2026-11-13)
--------------------

**Performance:**

* A Euclidean distance bound between a set of voxels and a set of peaks
  holds half the memory and takes about half the time: on 300 k peaks and
  the 318 k voxels of a 2 mm brain mask, a radius of 6 voxels (120 M pairs)
  peaks at 7.5 GB instead of 14.6 GB, and takes 18 s instead of 43 s, and
  a radius of 3 at 1.1 GB instead of 2.8 GB, in 1.6 s instead of 5.5 s. An
  integer join looked its neighbours up with the integer offsets of the
  radius only when it needed few look-ups, and built k-d trees otherwise,
  whose pairs are 5 to 10 times dearer than a look-up: it now looks them up
  whenever the larger set fills at least one cell of the grid of its
  coordinates in 40. The k-d trees are searched in chunks of the smaller
  set, so that the pairs they return are never all alive at once; and the
  distances, the 64-bit indices of the points and a copy of the pairs are
  no longer kept when they are not needed. Float coordinates lying on a grid
  (multiples of 1, 0.5, 0.25, 0.1, 0.05 or 0.01, as the millimetre
  coordinates of peaks and voxels are) are joined the same way, as integers,
  the pairs at the very radius being checked on the floats: they give the
  same pairs and distances as before. The thresholds, the sizes of the chunks
  and the other tunable values of the join are in
  ``neurolang/utils/euclidean_join.yaml``, read with OmegaConf (a dependency
  of ``neurolang`` now) and changed by the ``euclidean_join`` key of an
  engine (see :ref:`euclidean-join-settings`), with a ``memory`` profile, which the
  interactive REPL, in a terminal and in a browser, uses: it uses the
  stencil up to a billion look-ups, counts the pairs before storing them and
  searches the k-d trees in small chunks. The distinct pairs are decoded in
  blocks into 32-bit indices: a radius of 6 voxels peaks at 6.6 GB, and
  at 6.0 GB in the memory profile, instead of 14.6 GB.

* ``query_similarity(q, s)`` keeps the embeddings of the last 4,096 query
  texts and computes one dot product per call: a rule over many texts,
  ``cogat_definition(t, d), similar_study(d, s)`` over the 615 definitions
  of the Cognitive Atlas concepts, embedded a text again for nearly every
  row, as the rows come in any order and only the similarities of the last
  16 texts were kept.

* A rule with many negated atoms compiles in a time linear in them: each
  negation was joined with the difference of those before it, which put
  that difference in both operands and doubled the relational algebra with
  each negation (14 negations, as in the bundles of tract_querier
  translated to the vocabulary of the engine tracts, took most of an hour;
  now a
  second).
* The regions of a large atlas are found in a second rather than ten (1.5 s
  instead of 12 s for the 180 labels of FreeSurfer's 256³ ``wmparc``): the
  labels are searched for in the type of the image's values, all at once.

* A selection by a Python function of a single column, as
  ``startswith('17Networks LH SalVentAttn', l)`` on ``schaefer_activated``
  (291,533 rows) in a probabilistic rule, applies the function to the column
  instead of to every row: the probability of the Destrieux regions given a
  study activating a salience parcel of one hemisphere takes 2.4 s instead
  of 8.6 s, and of either hemisphere (a union of two rules) 12 s instead of
  37 s.
  
* The deterministic rest of a query that the disjoint separator joins with
  a probabilistic choice, as ``sub(t), mentions(s, t)`` in ``term_study(s)
  :- sub(t), mentions(s, t), selected_study(s)``, is computed as a set with
  the relational algebra, a union with a union and a projection, instead of
  with the operations on probabilities (sums, and a union of independent
  events through logarithms). The relational algebra optimiser flattens a
  projection of nested joins before pushing the projection into them, so
  that the joins are ordered by size; it joined the large relations first.
  The voxels activated by the studies mentioning a subclass of a Cognitive
  Atlas concept take 4.5 s instead of 6.2 s, and the Destrieux regions
  given a study activating a salience parcel of either hemisphere (a union
  of two rules with ``startswith``) 15 s instead of 37 s, and 2.3 s with
  the column-wise selection of #1126. ``--show-ra`` draws these relations as
  trees.
  
* Magic sets adorn first the body atoms holding a variable bound in the
  head, as they do the atoms with a constant: a query constant for a head
  variable, such as the text of ``p_a_given_similar('being socially
  excluded', i, j, k, p)``, restricts ``activated`` to the studies similar
  to it, as when the text is written into the rule, instead of joining the
  50 M activations of every study in the probabilistic plan: 0.9 GB more
  than the engine instead of 4.2 GB, and 23 s instead of 45 s, for the same
  1,151,590 rows.

* ``RESAMPLE`` sums the probabilities of the outcomes of the choice over all
  the replicates at once, as a product of matrices, instead of joining every
  replicate with every outcome: the 500 replicates of the probability that
  two Schaefer parcels are active given a term (``PROB[co_active(l1, l2) //
  term_association(s, 'auditory')]``) take 13 s and 1 GB more than the
  engine, instead of 43 GB, and run in the REPL in the browser.
  
* ``uniformity`` of a term (``uniformity('pain', i, j, k, u)``) takes 19 s
  instead of 55 s, and the memory of the engine instead of 18 GB, so that it
  runs in the REPL in the browser with the full engine (``?engine=full``):
  the chase projects a rule's substitutions on the variables of its head,
  which the optimisers push into the joins, instead of computing, and
  comparing with the rows already found, all the combinations of the body's
  variables (45 M rows of activations for 11 k of the head). A relation made
  from a data frame shares its columns, with copy-on-write, rather than
  copying them in one block, 1.4 GB for the activations.
  
* ``lift`` of a term (``lift('pain', i, j, k, l)``) runs in the REPL in the
  browser with the full engine too: a look-up join (the probabilities of a
  choice) and a grouped sum, maximum, minimum or count of many rows go a
  chunk of 4 M rows at a time, which for the probability of each voxel over
  45 M activations takes 0.5 GB instead of 1.2 GB and 0.2 GB instead of
  1.9 GB, for a second more.

**New features:**

* The meta-analysis engines (``neurosynth``, ``neurostore`` and those
  extending them) have the year, journal and abstract of each study,
  ``study_year``, ``study_journal`` and ``study_abstract``, next to
  ``study_title``, and its identifiers, ``study_pmid`` and ``study_doi``,
  the same in both engines, so that their studies are matched:
  Neurosynth's abstracts are PubMed's, NeuroStore's years, PubMed ids and
  DOIs those it lists for its studies, asked for at the first build and
  kept in the data directory. A query, and the query assistant, restricts the
  studies to years or journals. The SPECTER2 similarity of ``neurosynth``
  embeds its ``study_abstract``, as ``neurostore`` does. See *The
  publications of the studies* in the engine configuration documentation.
* The rest of the documentation starts from the REPL as well: the queries of
  the Cognitive Atlas page run on the ``neurostore`` engine from their link,
  with its counts in the 2026-11 release; Concepts writes facts and rules as
  the REPL takes them; the pages start the REPL as ``neurolang-query -i``;
  the README leads with the REPL in the browser.
* The SQUALL tutorial runs in the REPL too: its Bayes factor program is on the
  ``neurosynth`` engine and the Destrieux atlas, with the top 20 terms it
  gives, its aggregation examples on facts typed in the session and with the
  results they give, and its Python moves to :doc:`tutorial_python`, where
  it is tested. SQUALL has no ``#`` comments: the examples that had them use
  ``%`` lines.
* The Datalog tutorial runs in the REPL: its Bayes factor program is on the
  ``neurosynth`` engine and the Destrieux atlas, with the results and
  confidence intervals it gives, and its Python moves to a new page,
  *NeuroLang in Python* (:doc:`tutorial_python`), which also takes
  *NeuroLang for Logic Programmers* (its address now leads there). The
  page's examples are run by the tests.
* The documentation starts from the REPL: a page on the REPL, in the browser
  and in a terminal (:doc:`repl`), a Quick Start and a Get Started whose
  examples are queries of the ``neurosynth`` engine, and, under each example
  of a page, a link that opens the REPL in the browser with it in its box
  (``.. nl-repl::``). The examples are coloured as the REPL colours them.

* ``.view`` of a result with a value on the regions of an atlas (``ans(l,
  p) :- PROB[...]``) starts at the top 5% of its values, as a result of voxels
  does; regions alone, networks and streamlines start with all of them.
* The examples of the parcels the uncinate connects (``uncinate_cogat.dl`` of
  the engine tracts, its README and the assistant's example of
  ``neurosynth_tracts``) count the ends of the streamlines as the pairs
  ``count(x, e)`` of a streamline and which of its ends, rather than as the
  integers ``i == 2 * x + e``.
* The REPL in the browser starts with the engine Neurosynth + tracts
  (``neurosynth_iit``: Neurosynth with the tractogram and the bundles of the
  IIT atlas); Neurosynth and NeuroStore are in its select, and
  ``?engine=neurosynth`` starts with Neurosynth alone.  The bundle builds
  Neurosynth first all the same, whose text embeddings the tracts' engine
  shares (``DEFAULT_ENGINE`` of ``doc/make_repl_bundle.py``).
* The brain viewer of a result on the regions of atlases names the region
  under the mouse, and under the cursor, and draws the regions in the
  colours of their atlas (*atlas colours*), as a result of regions alone
  starts: Schaefer's (from its label file), FreeSurfer's for Destrieux
  (``FreeSurferColorLUT.txt``, fetched into the data directory) and Tian's
  (the label file of its authors).  The engine keeps the colours of its
  atlases (``engine_registry.atlas_colours``).
* When the AI assistant can answer, with a key in the page or the one of the
  REPL's configuration, the REPL in the browser starts with one of the engine's
  example questions, at random, in the box of the assistant: selected, typing
  replaces it, and *Ask* asks it.  ``WebRepl.assistant_ready`` tells the page.

* ``neurolang-repl-serve --local`` serves the REPL page with NeuroLang in the
  Python of this machine rather than in the browser: the engine is built
  from the data directory with all of its analyses (``lift('pain', i, j, k,
  l)`` answers in 9 s, where the browser's engine leaves it out), *Stop*
  interrupts the query on the server, files are made there, and the AI
  assistant is the one configured on this machine unless the page gives a
  key.  The page is the same; its worker, ``local_worker.js``, sends what it
  is asked to the server.
* ``neurolang-repl-serve --cert CERT.pem [--key KEY.pem]
  [--key-password-file FILE]`` serves over HTTPS (a key with a passphrase
  opened with that of the file, or asked in a terminal; a certificate that
  cannot serve says why), and ``--daemon install|reinstall|remove`` installs, replaces or
  removes a service of the user serving the REPL with the other options, at
  login: a unit of systemd's user manager, or a launchd agent on macOS.
* ``neurolang-repl-serve --password-file FILE`` asks every visitor for a
  password (HTTP Basic), ``--env-file ENV`` gives those with it, with
  ``--local``, the server's key of the AI assistant, and the site's main
  page leads to the REPL.  Without a password, the server answers only the
  requests sent to this machine's names, ``--host`` and those of
  ``--server-name NAME`` (against DNS rebinding), and, with a password or
  without, a POST only from the page of the site (in JSON, from its
  ``Origin``); a silent connection is closed after a minute, and a
  request's body is a JSON object of 1 MB at most.
* ``neurolang-repl-serve --users FILE`` asks a user's name and password
  (scrypt hashes, ``--hash-password``); with ``--local`` each tab of a user
  has a session in a process of its own, forked from a builder that builds
  each engine once, with the variables (the AI assistant's key) the file
  gives the user (``--max-sessions``, ``--idle-minutes``); a duplicated
  tab has a session of its own.  The feedback
  server's SQLite database is in WAL mode, for the writes of concurrent
  sessions.
* ``neurolang-repl-serve --users``: users have ``groups``, and those of
  ``admin`` the admin console at ``/admin`` (users, groups, the status of
  their assistant's key, sessions, ratings); with ``--local``,
  ``--feedback-db FILE`` stores the ratings of the page on the server with
  the user's name and groups, the rated answers of the assistant exported
  from the console as its examples, by engine.  A POST is taken from the
  server's own page only.  ``--add-user NAME --groups ...``,
  ``--set-password``, ``--set-groups``, ``--set-key NAME VARIABLE``,
  ``--remove-user`` and ``--list-users`` manage the users from the command
  line; the console and the command line write the users file holding a
  lock of it, and a password has 8 characters at least.
* With ``neurolang-repl-serve --users``, each user changes their own
  password on their account page, ``/account``, linked from the REPL's page.
* The users file of ``neurolang-repl-serve --users`` gives the groups their
  keys (``groups: {lab: {env: {...}}}``, beside ``users:``), which their
  users have unless they have their own; the console shows and sets them, as
  does ``--set-group-key GROUP VARIABLE``.
* A user or a group of ``neurolang-repl-serve --users`` has an assistant of
  its own, ``llm`` (a provider of the page's form, its model and key),
  which an admin sets in the console or with ``--set-llm`` and
  ``--set-group-llm``, and a user saves on their account page; ``own_key``
  (``--own-key``) lets them give their own key, or not, in the page and on
  their account page.
* ``neurolang-repl-serve --users`` has a login page, ``/login``, a signed
  login cookie and ``/logout``, which ends the user's logins made before,
  in every browser;
  ``--secure-cookie`` sends it over HTTPS only behind a proxy.  With
  ``--local``, the REPL shows the name of the user logged in, linked to
  their account page, and *Log out*, which forgets the API keys the page
  kept.  The page shows whether the server's assistant answers, and a key
  saved on the account page or in the console answers at once in the open
  pages of the users it changes.
* ``neurolang-repl-serve --guests`` lets visitors in as guests, from the
  login page: sessions of their own, without an account, whose assistant
  answers only with the key they give (no key of the server's, which is
  kept out of every session's process unless its user has it), at most
  ``--max-guest-sessions`` at once.
* ``.graph`` draws where the answer of the last query comes from, and how:
  the data of the engine, with their columns, its random choices and
  probabilistic facts, the rules of the query and of the engine, which are
  probabilistic, the probabilities ``PROB[...]`` and the answer, from left
  to right. A rule lists the atoms of its body, each where the edge of the
  relation it uses arrives, and says how it combines them: the variables
  it joins them on, those it computes, keeps, or drops. A probability says
  how it turns the random choices it depends on into a number for each
  row (with one uniform choice, the share of the choices where its body
  holds), and the answer shows the distribution of its probabilities. The
  colour of an edge tells what goes through it: rows that hold for sure,
  uncertain rows, or probabilities. The REPL in the browser shows it in a
  *Graph* tab after each query, a query of the assistant's included.
* The graph of a query draws the relations that come from an ontology (the
  triples and classes of the Cognitive Atlas, such as ``rdfs:label``) as
  teal hexagons, apart from the blue cylinders of the regular tables, and
  its legend says so. ``NeurolangPDL.ontology_predicates`` holds the names
  of the predicates the loaded ontologies define.

* A program holds several queries, each named in brackets,
  ``ans[name](...) :- ...``, and ``execute_datalog_program`` returns their
  answers in a dictionary by name. The queries are solved together: each is
  rewritten with magic sets for its own bindings, into one program that is
  run once, so that the predicates they share are computed once, for the
  bindings of all of them (four queries sharing a recursive predicate take
  1.6 s instead of 2.5 s). ``.view`` of the REPL shows the results of all the
  queries in one viewer: images, a network and streamlines, beside one
  another.

* One model of atlases, :class:`neurolang.atlases.Atlas`: a label image, a
  label table (``id``, ``name``, ``hemi``), a space tag and a background,
  with its loaders, the normalisation of label names, the hemisphere rules
  and its regions.  ``.load_atlas(name, ...)`` loads any atlas into
  ``name(label, region)``: ``kind="freesurfer"`` (a colour LUT),
  ``"bids"`` (a ``dseg.tsv``) or ``"tsv"`` from local files, the atlases of
  the engines (``kind="destrieux"``, ``"schaefer"``, ``"tian"``), a fetcher
  of nilearn (``kind="nilearn", fetcher="harvard_oxford"``), and, without a
  kind, an image and a CSV at URLs as before.  The engine records the atlas
  by name (``nl.atlases``), as do ``add_atlas_set``, which also takes an
  ``Atlas``, and the ``atlases:`` of an engine of ``engines.yaml``, whose
  relations are unchanged.  The ``tracts`` engine indexes a tractogram by
  any of them, ``.index_tracts(t, name, term_mapping=...)``, the term
  mapping defaulting to that of the atlas's kind (FreeSurfer, Destrieux,
  Tian); the engine's ``.load_tract_atlas`` is gone.  ``nl.atlas_labels`` is
  gone too: ``atlas_label_values`` reads the recorded atlas.

* The vocabulary of tractography (``vocabulary.dl`` of the engine
  ``tracts``) and the shipped translations of tract_querier's queries
  (its ``bundles``)
  document each predicate with a ``%%`` block: its signature with the sorts
  of its arguments, ``uf(x: streamline, h: hemisphere)``, and, for a bundle,
  its full name (``uf`` of ``freesurfer_queries`` is the uncinate
  fasciculus, from the engine's table ``tract_names.tsv``), a sentence on
  what it selects and the WMQL statement it is translated from.  The engine
  ``tracts`` declares the sorts (``streamline``, ``end``, ``term``,
  ``hemisphere``, ``axis``...) with descriptions, so that the query assistant
  lists ``uf``, ``cst``, ``cc_1``... by name, and ``.view`` draws a column of
  the sort ``streamline``.  The description of the engine names the bundle
  modules, so that the assistant imports one
  (``.import('bundles/freesurfer_queries.dl')``) to answer with
  ``uf``.

* Tractography is the engine ``tracts`` of the apps,
  ``neurolang_apps.utils.engines.tracts``, rather than the workspace package
  ``nl_tracts``, which is gone: its modules, vocabulary, bundles, mappings,
  examples and tests are those of the engine, its tools in
  ``packages/neurolang_apps/tools/tracts`` (``wmql_to_datalog.py``, the
  former command ``nl-tracts wmql``), and the extra ``tracts`` needs
  nothing more than the apps.  ``frontend`` keeps the solver's mixin of the
  commands and the ``python_init``, which imports the vocabulary; an engine
  of the apps has them by extending ``tracts``.  ``register``,
  ``relation_frames``, ``RELATION_NAMES``, ``TractsSolver``,
  ``tracts_engine`` and the module ``vocabulary`` are gone
  (``vocabulary_predicates()`` lists the predicates of the vocabulary).

* The vocabulary of tractography can reproduce tract_querier where the two
  differ by design (the engine ``tracts``, DESIGN.md 6.1, 6.1.1, 17.5).  With
  ``.index_tracts(..., emit_background=1)`` (as before: rows of the
  background label in ``streamline_label``) ``only_in``, hence ``only(...)``
  of a translated query, is tract_querier's, which counts the background
  among the labels of a streamline; the new option
  ``.index_tracts(..., extent="centre")`` (``label_extent_relation(atlas,
  extent)``, ``RelationStore.add_atlas(atlas, term_mapping, extent)``) gives
  the boxes of the labels by the centres of their voxels, as tract_querier's
  without scipy, instead of the voxel edges, the default, so that the
  directional terms (``anterior_of``...) are its own.  On the whole brain of
  HCP842 and ``wmparc``, the 58 bundles of ``freesurfer_queries.qry`` (both
  sides) are the same streamlines as tract_querier's with both options (50
  without) and the 43 translated of ``freesurfer_queries_bsf2016.qry`` and
  of its ``_toad`` variant, too.

* The query assistant votes with ReAct too: ``samples: N`` answers N
  times and keeps the answer whose rows most of them give, and
  ``fallback_provider: NAME`` asks another provider once when no result
  has a majority (more than N/2 of the answers) or none is valid, e.g.
  three DeepSeek answers and Sonnet when they disagree. The fallback is a
  step of the answer, with the query the vote kept.

* A block of ``%%`` lines right before a rule or a fact of the Datalog
  syntax documents its predicate, with an optional signature giving the
  names and the sorts of its arguments, ``%% uncinate(x: streamline, h:
  hemisphere)``, then its description (see :ref:`datalog-doc-comments`).
  The engine keeps it, in the scopes of the program, with the modules a
  program imports (``nl.predicate_documentation()``,
  ``nl.document_predicate()``, ``.document(name, "description")``); the
  query assistant tells the model the description and the sorts of a
  documented predicate, before those of the engine's configuration, as do
  the REPL's ``.predicates`` and ``.datalog PREDICATE``,
  ``neurolang-query --list-predicates -f program.dl`` and the predicates of
  the REPL in the browser; ``.view`` draws a column documented of the sort
  ``streamline`` as streamlines. The pretty printer writes the blocks back,
  and a ``.command(...)`` as written. The model is told the sort of the
  argument of a declared predicate too, ``reports(study_id: study, ...)``,
  and the description of each sort of an argument that the engine declares.
* Datalog modules: ``.import("bundles/uf.dl")`` in a program runs the facts,
  rules and commands of a Datalog file where it stands, as if written there,
  so that a vocabulary or a set of definitions (the white-matter bundles of
  a tractography package) is written once and imported by queries.  The
  path is relative to the importing file (the program of
  ``neurolang-query --file``, or a module importing another), else to the
  working directory; ``.import("package:bundles/uf.dl")`` imports a module
  shipped in a Python package.  A module is imported once (importing it
  again does nothing), an import cycle is an error, and a module with a
  query (``ans(...)``) is refused.  An error in a module names its file,
  and its line if it does not parse.  From Python:
  ``nl.import_datalog_module(path)``.  In the REPL, what a query's imports
  define stays in the session although the query's own rules are local to
  it, and ``.clear`` forgets the imports with their rules.
* ``.import`` looks for a module not found relative to the importing file
  or the working directory in the directories the engine gives its modules,
  ``module_directories`` of its solver: the engine ``tracts`` gives its
  own, so that a program imports its bundles as
  ``.import("bundles/freesurfer_queries.dl")``.
* A rule body of the Datalog syntax may be a disjunction, ``|`` or ``∨``,
  which binds looser than ``,``; parentheses group: ``uncinate(x) :-
  (traverses(x, 'orbital') | traverses(x, 'middle')), temporal(x)``, rather
  than a predicate of several rules.  **Breaking:** ``|`` is no longer a
  synonym of ``//`` for conditioning, ``PROB[a(x) | b(x)] = p`` is the
  probability of the disjunction; write ``PROB[a(x) // b(x)] = p``.  A
  comparison whose left operand is a function call in parentheses,
  ``((f(x)) < 3)`` or ``((f(x), y) == (1, 2))``, now reads as a formula in
  parentheses and fails to parse; write ``(f(x) < 3)``.
* The brain viewer of a network (``.view`` of a result of pairs of regions or
  points) thresholds its edges in the page: "Edges from" and its slider draw
  the network again with the edges above the value, the strongest 5% at
  first. An edge is thicker with its value from the threshold, where it is a
  fifth of the thickest, to the largest value, rather than in proportion to
  the value, which made edges of close values look alike.
* The REPL in the browser asks "Did this query do what you wanted?" after a
  query typed in the page too, not only after an answer of the assistant, in
  a row between the output and the result: a rating
  sends the query with its first rows or its error, and the comment, to the
  same Grist table, with an empty question.
* In the REPL in the browser, the examples and relations the page opens on a
  first visit close at the first run, so that the result is in view, and a
  query that fails without a result hides the last result, not to be taken
  for its own.
* The REPL in the browser asks whether an answer of the assistant was what
  the visitor wanted, and sends it, rated good or bad with a comment, as a
  row of a Grist document that anyone can add rows to (``feedback.json`` next
  to the page names it; without a document the page does not ask). Nothing of
  an answer is sent unless it is rated, and the page holds no key: Grist does
  not accept one from a web page.
* The relations and rules list of the REPL in the browser leaves out the IRIs
  of an ontology's classes and properties, such as
  ``http://www.cognitiveatlas.org/ontology/cogat.owl#CAO_00137``, unless the
  engine declares them or the session defines them; they can still be used in
  a query. The relations named with a prefix, ``rdfs:label``, stay.
* The LLM assistant can record its questions and answers in a Grist document
  instead of a feedback server (``grist_doc`` in the ``feedback`` settings,
  the API key in ``GRIST_API_KEY``), one row of the table ``Answers`` each, to
  be flagged good, bad or unsure in Grist or with ``.rate`` and ``.refine``.
  The answers that failed for an error of the provider are shared too.
* The browser REPL opens its examples on the first start, when the box is
  empty; the assistant's *Ask* button is filled only once a key is in use, so
  that *Run* is the only primary action; the empty gap above the examples is
  gone; a result table is as wide as its columns, with the header over its
  numbers; and the keyboard focus is visible on the lists and the sortable
  headers.

* The browser REPL: the query can be written while the engine loads, which
  says how long it takes; a failed start says in plain words what to do, with
  a *Try again* button; the Brain tab is shown only when there is a brain to
  see; the log of a query is folded to the time it took; the result tabs are
  a tab list for the keyboard and screen readers; the table's controls are
  shorter and the page fits a phone; and the button for the assistant's key
  opens the key form, tells when a key is in use, and says what *Ask* needs.

* The browser REPL shows the probabilities of a result (a ``p`` or ``prob``
  column of numbers between 0 and 1) as bars, the most probable rows first.

* The browser REPL has a *Use in a query* button on a result: the query that
  made it is put in the box again, its answer renamed ``result``, with a new
  ``ans(...) :- result(...), `` to complete. (A result is not a relation of
  the session: ``ans`` cannot be used in a body.)
  A result of studies also has *Map the studies*: the query that counts, for
  each voxel, the studies of the result that report a peak there, which the
  brain tab shows.

* The browser REPL has *Ask a question* in its examples: a form for the common
  questions (the studies that mention a term, two terms, one and not the
  other, a term a lot; their peaks; the terms of a study), the ones the
  engine has the predicates of, which puts the Datalog query in the box.

* ``.explain`` reads the last query that ran in English, with the assistant;
  the browser REPL has an *Explain* button for it once an API key is in use.

* In the browser REPL, the studies of a result are links: to their PubMed
  record, to their DOI (the engine's metadata have it; the tooltip gives the
  title, authors, year and journal) and, when Europe PMC knows it, to their
  PubMed Central record. Europe PMC is asked for the PubMed ids of the studies
  on the page, nothing else; if it cannot be reached the other links stay.
  A table tells the sort of each of its columns (``sorts``) and, if some are
  studies, their ``study_links``.

* The browser REPL documents each relation and each command. In *Examples and
  relations*, the button *ⓘ* of a relation tells what it holds (the engine's
  description), its columns with their sort and type, and where it comes from,
  and puts it in the box or the query for all its rows; there is a list of the
  commands, each with what it does and its full help. One filter searches both
  lists. ``WebRepl.predicates()`` gives the ``arguments`` of each relation and
  ``WebRepl.commands()`` the commands.

* The browser REPL has a *Stop* button while a query runs: the engine is
  restarted (its data are kept by the browser) and the rules defined in the
  session are run again; *Save session* and *Open session* keep the
  statements of a session in a file and give its rules back.

* ``uv run --extra apps neurolang-repl-serve`` serves the REPL in the browser
  of a checkout on this machine: it builds the page's bundle the first time
  (or with ``--rebuild``) and serves it uncached at
  ``http://localhost:8766/_static/repl/index.html``. It is the command of the
  workspace's root project, whose only package is :file:`tools/serve_repl`.
* ``query_similarity`` works in the REPL in the browser. Without ``torch``,
  SPECTER2's query encoder runs in numpy (``neurolang.utils.specter2_numpy``:
  BERT's WordPiece tokenizer and the model with its ad-hoc query adapter),
  giving the embeddings of ``torch`` to within float16 rounding, with the
  same ranking of the studies. The page fetches the exported model (268 MB,
  in parts under GitHub's 100 MB) and the embeddings of the studies (44 MB)
  the first time a query needs them; a query then takes about 2 s.
  ``text_embeddings.fetch_missing`` fetches the files the embeddings lack.

**Internal:**

* Every pull request gives its reviewer the REPL in the browser of its
  branch: the documentation's workflow uploads the site (the documentation
  and the REPL) as the artifact ``repl-pr-NUMBER`` and keeps one comment of
  the pull request up to date with the commit it was built from, the
  artifact, whether the REPL's bundle is complete and how to run it:
  ``uv run --extra apps neurolang-repl-serve --pr NUMBER`` (or ``--run-id
  ID``), which downloads it with the GitHub CLI and serves it without
  building anything, or ``gh run download`` and ``python3 -m http.server``.
  A pull request from a fork has the artifact, without the comment.
* The workflows of the tests, the layers, the spelling and the
  documentation run on every pull request, whatever its base, so that a
  pull request stacked on another one is checked (its tests are those of
  the packages it changes with respect to the pull request below it); a
  push to a pull request cancels its runs still going.
* A REPL's bundle that could not be built fails the check *REPL bundle*
  instead of passing unseen (the documentation is deployed all the same).
  ``doc/make_repl_bundle.py --downloads DIR`` keeps what the engine
  downloads in DIR, even when the build fails, and CI keeps that directory
  from one build to the next; a download failing for a while (a time-out,
  an error 5xx) is tried again after 2, 4 and 8 s
  (``neurolang.utils.nilearn_compat.fetch_files``).
* Helpers reused instead of repeated: the conjuncts of a formula and the
  selection of a plan by comparisons in the probabilistic lifting, the
  aggregate types of the magic sets, the pandas reductions table, the
  Euclidean join's index type, the export, the insert, the ``extends`` and
  the missing optional dependency of the apps; docstrings of the apps name
  ``neurolang_apps``.
* The engine configuration is composed with OmegaConf (``omegaconf>=2.4``,
  a new dependency of ``neurolang_apps``, pure Python, installed in the
  browser's REPL too): the example queries, the query assistant's bank of
  examples and its guidance of an engine are in
  ``engines/<engine>/examples.yaml`` and ``engines/<engine>/guidance.yaml``,
  merged into its entry of ``engines.yaml``; the banks
  ``llm/examples/<engine>.yaml`` become the engines' ``assistant_examples``,
  inherited through ``extends`` like ``builtins``, and
  ``neurolang-feedback-server export --format examples`` writes them under
  that key. ``extends`` merges mappings with ``OmegaConf.merge``;
  ``engine_bases`` is removed. The resolved configuration of every engine
  is unchanged, and read 6 times faster after the first time.

**Bug fixes:**

* The radius of the activation of a study around its peaks is one
  ``kernel_radius_mm`` of the meta-analysis engines, 6 mm, the same at every
  resolution: ``activated`` was within 3 voxels (3 mm on the default 1 mm
  grid, 6 mm on the 2 mm one), and the atlases (``destrieux_activated``...)
  and the assistant within 3 mm.  ``activated`` takes it in voxels of the
  grid (``{kernel_radius_voxels}`` in ``datalog_init``), the atlases as
  their default ``radius_mm``, and the descriptions, the guidance and the
  examples of the assistant where they say ``{kernel_radius_mm}``; the
  banner of the REPL gives it with the size of the voxels.  On the 2 mm grid
  ``activated`` is unchanged; on the default 1 mm grid it is 6 voxels
  instead of 3, about 455 million (voxel, study) pairs of Neurosynth's
  peaks instead of 61 million, at least 14.6 GB instead of 2.0 GB for its
  four 64-bit columns: use ``-r 2`` for a smaller engine.  The atlases'
  activations, within 6 mm instead of 3, change (#1070).
* An atlas lookup with a voxel, ``ans(l) :- destrieux(l, r) &
  contains(r, (68, 77, 37))``, gives the labels of the regions holding it
  instead of an internal error (``TypeError: keywords must be strings``):
  ``contains`` with a tuple of constants is a membership test, not the
  explosion of the set into columns named by the constants, which only a
  tuple of variables, ``contains(r, (i, j, k))``, is (#1074).
* The atlases of an engine with a template (Schaefer, Destrieux, Tian) are
  on its grid: their label images are resampled to the template by nearest
  neighbour when they are loaded (and kept in the engine's cache, by the
  grid of the template), so that the voxel (i, j, k) of ``labels`` and of
  ``contains(r, (i, j, k))`` on an atlas region is that of ``voxel`` and
  ``reports``.  With ``-r 2``, the template voxel (68, 77, 37), MNI (38,
  20, 2) in the right insula, was labelled by Schaefer's 2 mm FSL grid as
  a parcel of the left hemisphere (``17Networks LH DefaultB PFCv 2``); it
  is now ``17Networks RH SalVentAttnA Ins 1``, and ``R G insular short`` of
  Destrieux.  The points of the atlases (``*_points``) and the studies
  activating their regions, in millimetres, stay those of the atlases'
  own grids (GH-1075).
* A string constant spelled as a variable is a value, not that variable's
  column (#1086): ``h(u) :- act(u, y) & (u == 'u')`` keeps only the rows of
  ``act`` whose first column is ``'u'``, instead of every row, in
  ``NeurolangDL`` and ``NeurolangPDL``, and ``h(u, z) :- act(u, y) &
  (z == 'z')`` gives ``z`` the value ``'z'`` instead of failing. A constant
  is equal to a column of the same value (``Constant[str]('u') ==
  Constant[ColumnStr]('u')``): the translation to relational algebra dropped
  such an equality as trivially true, and an extended projection
  ``'z' → z`` was read as keeping the column ``z``, on provenance sets and
  when moving computed columns above a join.
* A query of the probabilistic frontend binding a variable with a constant,
  as ``d`` in ``cogat_definition('working memory', d)``, next to a literal
  sharing no variable with the others, such as a second ``PROB[...] = pa``,
  is rewritten with magic sets again: walking the adorned rules reordered
  their bodies to avoid cross products, putting the literals bound by ``d``
  first, and the magic sets created from the reordered bodies left those of
  the adorned literals undefined. The query raised ``UnboundVariablesError``
  on a similarity left unbound, or answered nothing (#1091).
* The Destrieux atlas of the engines is on the MNI152 template: nilearn's,
  derived from FreeSurfer's fsaverage, is 1.3 to 2.6% smaller than the
  template and off by about a millimetre, and is now moved by the scales and
  shifts fitted to the template's grey matter (``DESTRIEUX_TO_MNI152`` of
  ``neurolang.atlases``, ``tools/fit_destrieux_mni152.py``): the
  grey-matter probability of its cortex goes from 0.627 to 0.649, that of
  Schaefer's being 0.636.  Its voxels are unchanged, its affine is moved;
  the studies activating its regions (``destrieux_activated``) are found
  again, and change a little.
* The streamlines of a tractogram in MNI space (``neurosynth_iit``'s IIT
  tractogram) are drawn on the MNI152 T1, not on the labels of the atlas
  they are indexed by: the engine's ``streamline_background()`` gives that
  of a subject's space only, None in MNI space, and the viewer then draws
  them on its template, without warning that they may be misplaced.
* The brain tab of the REPL in the browser no longer shows the viewer of an
  earlier result for the last one: after a query, selecting the tab views
  the new result (``.view``), and a view that fails ("The result has no
  streamlines") takes the earlier viewer away rather than leaving it, as
  a second query on tracts often did.
* A query on tracts (``ans(x) :- af(x, "left")``) no longer prints notes about
  predicates of the magic-sets rewriting (``endpoint_in^bfbb_0 has no facts or
  rules``): the types of their rules are not resolved, as for those whose
  functor is an ``AdornedSymbol``.
* ``count(s)`` counts the distinct values of ``s`` in each group, not the
  rows of the rule's body (#1090): ``n(count(s)) :- schaefer_activated(l, s),
  startswith('17Networks LH Default', l)`` is 11,160 studies, not 50,655,
  one per parcel each activates. ``count(x, y)`` counts distinct pairs; the
  other aggregations still take a value per row. In SQUALL too, and with
  the polars backend, where an aggregation of several columns without
  groups failed.
* A comparison inside parentheses is a disjunct or conjunct of its own:
  ``ans(i, t) :- cogat_term(i, t), (t == 'motor execution' | t == 'motor
  planning')`` parses; before, every comparison of the disjunction needed
  parentheses of its own, ``((t == 'a') | (t == 'b'))``.
* In the REPL in the browser, the Brain and Graph windows are as tall as
  their page, instead of 34 rem with scrollbars of their own within the
  page's (the readout and the colour bar of the viewer were cut off), and
  they follow the light or dark theme of the REPL, switching with it: the
  brain viewer was always dark, drawing the image on black, and the graph
  followed the system's theme, not the REPL's. In the light theme the
  viewer draws the image on white. Opened apart, as ``.view`` does in the
  terminal, they follow the system's theme as before.
* The query assistant's ``anthropic`` provider, in the terminal and in the
  browser REPL, no longer sends ``temperature``, which current Claude models
  reject (HTTP 400 "`temperature` is deprecated for this model"). It sends
  back each reply's content blocks unchanged, so the thinking blocks of
  Claude Sonnet 5.5 stay with its tool calls, and gives it 16000 output
  tokens since its thinking counts in them.

* The documentation's CI builds the REPL in the browser for ``master`` only,
  with the Neurosynth engine only, and keeps the embeddings of its studies
  between builds, saved whenever a build changed them: a build embedded the
  studies again each time, an hour for Neurosynth's and four for
  NeuroStore's, and some reached the six-hour limit of a job.  A pull
  request's CI uploads its documentation without a REPL that can start, and
  no longer comments on how to run it; ``neurolang-repl-serve`` builds the
  REPL of a branch locally.

* ``_`` in a negated atom or in a disjunct of a rule body is quantified
  there: ``~traverses(x, _, h)`` is "no term", ``~∃t traverses(x, t, h)``,
  instead of an error that the rule is not safe.
* ``.load_atlas`` put a tuple of the region, not the region, in the second
  column of its relation, ``Tuple[str, ExplicitVBR]``, and kept the labels
  without voxels.  The background (label 0) is no region any more either.
* Several probabilistic facts of a predicate written in a Datalog program
  (``0.5 :: p(1)`` and ``0.2 :: p(2)``) made a relation without the type of
  its columns, and a query over them failed with an ``IndexError`` when its
  arities were checked.
* The Neurosynth data are downloaded from raw.githubusercontent.com:
  github.com/neurosynth/neurosynth-data/raw/master/, which sends there,
  answers 504 for that repository, so that the ``neurosynth`` engine could
  not be built from nothing, nor the REPL's bundle, whose build passed
  unseen and left the deployed page unable to start.
* The semi-naive chase merged, at every step and for every rule, the
  relation of each predicate of the step's delta with that delta, even
  the predicates the rule does not use, and inferred the type of the
  merge by iterating over it: on some evaluation orders (they depend on
  ``PYTHONHASHSEED``) a query over a whole-brain tractogram took minutes
  and 45 GB instead of 3 s.  Only the negated predicates of a rule are
  merged now.
* The lifted probability of a union, as ``q(x) :- a(x)`` and ``q(x) :-
  b(x)``, merged a tuple of ``a`` and ``b`` with the same probability into one
  event: two facts of probability 0.5 gave 0.5 instead of 0.75.
* A negation written ``¬a(x)`` in the Datalog syntax failed to parse.
* Magic sets leave unbound a predicate whose binding would put an
  aggregation or a negation in a cycle of the rewritten program, and bind
  the others: a query over the vocabulary of tractography, whose
  ``term_box_hi`` aggregates, was not rewritten at all, and a function of a
  variable it bound, as ``query_similarity`` in ``similar_study``, was
  refused as unbound.

* Magic sets rewrite a query without a constant when a rule binds the
  argument of another right of it, as ``d`` in ``ans(t, s) :-
  cogat_definition(t, d), close(d, s)``: a function of ``d`` in ``close``
  was refused as unbound.

* A Datalog rule whose head starts with a constant, ``n('VisCent', l) :-
  schaefer(l, r), ...``, failed to parse: the parser read it as a fact
  and stopped at the variable.  A fact is now parsed as a head and checked
  to have only constants (``p(x)`` alone says that a ``,`` or ``:-`` may
  be missing), so completion after ``A(3)`` also offers ``:-``.
* The ``neurosynth`` engine converts only the peaks of Talairach (``TAL``)
  studies to MNI: those of ``UNKNOWN`` space, 53,878 peaks of 1,832 studies,
  were converted too, moving them by 7 mm (median).  They are now taken as
  MNI, as Neurosynth and NiMARE do; the engine option ``unknown_space``
  (``python_init_options`` in ``engines.yaml``) converts them as Talairach
  (``tal``) or leaves them out (``drop``) (#1071).  The ``neurostore``
  engine, which left out the peaks of any other space, now follows the
  same option: they are taken as MNI by default.
* The embeddings of the studies (``query_similarity``, ``similar_study``)
  are kept whatever the order of the study ids: their cache was keyed on the
  ids in the order of the ``study`` relation, which the fix above changed,
  so every free-text query embedded the 14,371 abstracts again and, in the
  query assistant, was stopped by its time limit before the embeddings
  were saved.  The ids are now sorted; the cache is rebuilt once.
* In ``NeurolangPDL``, a rule whose body is only equalities to constants,
  ``p(l) :- (l == 'c')``, is the fact ``p('c')``: it raised an
  ``IndexError`` (#1076).  The equalities to constants that the rewrite
  inlines are now only those of a variable with one of them and found
  elsewhere in atoms alone: ``(v == 'a') & (v == 'b')`` gave ``v = 'b'``,
  ``(v == 'b') & (w == v)`` dropped the constraint on ``w`` and
  ``(v == 'b') & (v != 'c')`` raised an ``UnboundVariablesError``.
* ``.view`` (the Brain button of the REPL in the browser) of a result over
  ``RESAMPLE``, ``ans(l1, l2, mean(v)) :- rep(b, l1, l2, v)``, knows the
  sorts of its columns: those of the predicate resampled, here Schaefer
  parcel labels, and so draws it as a network rather than failing with "The
  result has no voxel columns".
* ``neurolang-repl-serve --rebuild`` serves the wheels it built, not those of
  an earlier build: their versions follow the commits, so the old ones kept
  their names next to the new ones and could be the ones the bundle named.
  ``--clear`` removes the bundle built before, then builds it again.
* The REPL in the browser could not start: its page and the script that builds
  its bundle (``doc/make_repl_bundle.py``) imported modules from
  ``neurolang.utils`` that the split into packages moved to
  ``neurolang_apps.utils``.
* The union of an instance known to be empty and one known not to be was
  taken for empty, which could end a semi-naive chase before its fixpoint.
* ``PROB[...]`` and ``MARG[...]`` accept their conjunctions in parentheses,
  as a rule's condition does: ``PROB[(a(x, s), (x < y)) // (b(s), c(s))] = p``
  was a syntax error.
* The REPL in the browser, whose engine leaves out the analyses that need
  gigabytes (``activated``, ``uniformity``, ``lift``... and the rules
  defined from them), says so: its banner lists them, its examples and the
  examples of its assistant leave out those using them, the assistant is
  told never to use them, and a query using one fails with "is not
  available in the browser" and the command of the REPL that has them,
  rather than with an unknown predicate or an empty result, in the REPL and
  for the assistant (#1068).

**Performance:**

* Recursive rules: from the second iteration on, the semi-naive chase joined
  the whole relation again and re-evaluated the rules without a recursive
  atom. A rule with one recursive atom now has it bound to the tuples new
  since the previous iteration, and is skipped when there are none. A
  transitive closure of 2.56 M tuples takes 21 s and 1.8 GB instead of 49 s
  and 2.5 GB with pandas, and one of 10 M tuples 104 s instead of more than
  120 s. Non-recursive queries are unchanged.

* A probability over the choice of a study with a peak within a radius of
  each voxel, ``PROB[brain_mask(i, j, k), (euclidean(i, j, k, i1, j1, k1) <
  3), peak_reported(i1, j1, k1, s), selected_study(s)]``, takes 13 s and half
  the memory, instead of 47 s and close to 10 GB, and now runs in a browser
  tab. The natural join with a relation unique on the only common column, as
  the probabilities of a choice are, is a look-up rather than a pandas merge;
  Python's ``sum``, ``max``, ``min`` and ``len`` in an aggregation run as
  pandas reductions on columns of numbers without missing values; groups are
  not sorted; and the Euclidean join keeps no distances it is not asked for,
  numbers its pairs of rows with 32-bit integers and does not copy its result.

**New features:**

* Confidence intervals by resampling the studies.
  ``nl.add_resampling("boot_studies", choice="selected_study")``, or
  ``resamplings`` in an engine's YAML, adds replicates of a probabilistic
  choice, the Bayesian, classical or Poisson bootstrap of its outcomes.
  ``RESAMPLE[boot_studies](pred(x)) = b`` is then ``pred(x)``, any rule
  over ``PROB`` of the choice including the measures computed from the
  probabilities, such as Bayes factors, on replicate ``b``, and
  ``percentile(x, 2.5)`` and ``percentile(x, 97.5)`` over the replicates
  give the interval. Every query sees the same replicates, so that two
  measures are compared on the same ``b``. It rewrites the rules into
  sums over the outcomes of the choice, weighted by each replicate, and
  needs no other solver; on Neurosynth, the interval of the Bayes factor
  of a term for a region takes 15 s. The ``neurosynth`` and
  ``neurostore`` engines have ``boot_studies``, 500 and 200 replicates of
  ``selected_study``, and the query assistant has examples and guidance
  for questions on how certain an answer is. See the tutorial,
  :doc:`tutorial_datalog`, section 10.7.
* An engine is built from the tables it kept, instead of finding them again.
  The studies activating the regions of its atlases took 48 s of the 58 s of
  the ``neurosynth`` engine's build, and nilearn, imported to find the MNI
  mask, the brain and cortex voxels and the atlases, more than a second
  (several in a browser). They are now kept in the data directory, in 3.6 MB
  of ``.npz``, ``.nii.gz`` and ``.json`` files that the terminal REPL and the
  REPL in the browser both read, without importing nilearn or scikit-learn:
  the build takes 6 s natively instead of 58 s and 8 s in Pyodide instead of
  112 s, and the page starts in 23 s instead of 127 s. The documentation
  workflow builds the engine for its bundle, so that the tables are in its
  data zip, and stops if none was kept. ``python -m
  neurolang_apps.utils.engine_cache ENGINE`` keeps them, lists them
  (``--list``) and removes them (``--clear``), and ``neurolang-query
  --no-engine-cache`` builds the engine without them. nilearn's file fetcher
  returns the files already in the data directory without being imported.
  The page says which step of its start it is in (importing NeuroLang,
  loading the Neurosynth data, the atlases, the ontologies...) instead of
  "Building the engine" for all of it: ``build_engine`` takes a ``progress``
  function that is told each step before it starts.

* The REPL in the browser shows the work of the AI assistant as it happens, in
  a section of the page you can open: the model's calls, the tool calls of
  the ReAct agent, its queries and what they gave back. The code box colours
  the query as the terminal does, in Datalog and in SQUALL. Each result has a
  Download menu: all its rows as CSV or TSV, and its NIfTI image, the file
  ``.save`` writes, for each value column; the brain tab offers the images
  too. In the terminal REPL and in the page, a *Summary* of the query the
  assistant wrote, its explanation and the query read back in English, is
  above the query, apart from it, instead of below it. The ReAct agent runs
  its final query to accept it, and the result it got is shown right away,
  in the terminal and in the page, instead of the query being run a second
  time: Enter on the query as it is shows that result again, and an edited
  query is run as usual. A result larger than 5 million cells, or a query with
  ``.scope off``, is run when asked, as before. The *Docs* menu and a hint under the box link the keyword at the
  cursor (``PROB``, ``AGGREGATE``, ``:-``…) to its section of the tutorial;
  the examples and the relations of the engine, and the rules you define,
  are listed, to put them in the box with a click; and *Copy link* shares the
  query in the address (after the ``#``, which no server receives).

* The example LLM configuration has the providers ``albert-deepseek``,
  ``openrouter-gemma`` and ``openrouter-deepseek``: DeepSeek V4 Flash and
  Gemma 4 31B on Albert and on OpenRouter.

* The query assistant can share its questions and answers with a feedback
  server, where people rate them correct, incorrect or unsure and correct
  their query, to train models and grow the banks of examples. It is off
  by default: ``feedback`` in ``llm.yaml`` or
  ``NEUROLANG_LLM_FEEDBACK_URL`` turns it on. In the REPL, ``.rate`` and
  ``.refine`` rate and correct the last answer. ``neurolang-feedback-server``
  (no new dependency) collects them, shows a review page and exports
  them as JSON lines or as a bank of examples. See
  :ref:`assistant-feedback`.

* The query assistant searches the studies by a description or a
  Cognitive Atlas concept: the ``neurosynth`` and ``neurostore`` engines
  share a bank of examples (``coordinate_meta_analysis.yaml``, which banks
  now inherit through ``extends``) with questions answered with
  ``similar_study``, ``p_a_given_similar``, ``uniformity_llm`` and the
  new ``cogat_definition(term, definition)``; ``neurostore.yaml`` adds
  questions on its LLM-extracted concepts and domains.  ``neurosynth``
  gets ``study_title(study, title)``, from its metadata, as ``neurostore``
  had.

* The engine of the REPL in the browser leaves out the precomputed analyses
  (``lift``, ``relative_risk``, ``uniformity``, ``ratios``): building them took
  almost 4 GB, which Safari does not give a tab, and made the page loop on
  "Building the engine". The page builds them with ``?engine=full``.
* The page of the REPL in the browser has a computer, light or dark theme
  switch, says when the browser stops its worker, shows how long a slow step
  lasts, and reads ``bundle.json`` afresh, so that a new deploy is not met by
  the name of the old wheel.
* NeuroLang is split into four packages, in a `uv workspace
  <https://docs.astral.sh/uv/concepts/projects/workspaces/>`_ under
  :file:`packages/`: ``nl-expr`` (the expression engine, imported as
  ``nl_expr``), ``nl-core`` (relational algebra, Datalog and probabilistic
  Datalog, ``nl_core``), ``neurolang`` (the languages, frontends and brain
  regions) and ``neurolang-apps`` (the ``neurolang-query`` command line, its
  interactive mode, the query assistant and the engines,
  ``neurolang_apps``). ``neurolang[apps]`` installs them all; ``neurolang``
  alone is the library without the applications. The modules keep their
  names under ``neurolang``, which are the same modules:
  ``neurolang.datalog`` is ``nl_core.datalog`` and ``python -m
  neurolang.utils.cli`` still runs the command line. The tests of a package
  use only that package and those before it, and CI runs the tests of the
  packages a pull request changes, on every Python version, and those of the
  packages built on them, on Python 3.12: a change to the terminal interface
  no longer runs the tests of the engine. A clone installs with ``uv sync
  --all-packages --all-extras --no-extra embeddings``, or ``uv pip install
  -e ".[apps,dev,doc]"``; see :doc:`contributing`.

* ``neurolang-apps`` depends on ``pyarrow``, which the ``neurostore``
  engine needs to read the parquet files of its releases and which
  NeuroLang never declared.

**Bug fixes:**

* A function of the engine, such as ``query_similarity``, and a threshold on
  it within ``PROB[...]`` give the same probabilities as the deterministic
  rule computing them, instead of an error: the program of the
  probabilistic stratum has the engine's functions, and the deterministic
  relation replacing the function of an atom resolves them.

* Two aggregations with arguments in one head, such as
  ``percentile(v, 2.5)`` and ``percentile(v, 97.5)``, gave the result of
  the last one for both: their functions shared one global name.

* A query whose head aggregates over probabilities, such as
  ``ans(max(p)) :- p_term(t, p)``, failed with
  ``NoValidChaseClassForStratumException``: it was solved by the
  deterministic chase, which does not compute the probabilities.

* A probability over a predicate of two rules, one of which binds a
  variable of the head only through an atom both share, as
  ``a(s, r) :- region(r), selected_study(s), activates(s, r)`` and
  ``a(s, r) :- region(r), selected_study(s), mentions(s, 't1')``, failed
  with ``NeuroLangTypeException``: the lifted plan was the union of the
  relations of ``activates(s, r)`` and ``mentions(s, 't1')``, without the
  same columns. Such a union is declined, see
  :doc:`probabilistic_lifting`.


v0.1a21 (2026-10-21)
--------------------

**New features:**

* The page of the REPL in the browser is a small interface: a box to ask
  the AI assistant, a box for the Datalog or SQUALL code with its history
  in a dropdown, and the result as a table or in the brain viewer.
* The REPL in the browser has the AI assistant (``.ask``, ``.llm``) with
  the visitor's own API key, given in the page for Anthropic, OpenAI,
  OpenRouter or any OpenAI-compatible server. The key stays in the tab
  (or on the device, if asked), goes only from the page to the provider,
  and the query the assistant writes is put in the input. The LLM clients
  send their requests with the browser's ``XMLHttpRequest``
  (``neurolang.utils.llm.browser``), and queries run inline where there
  are no threads nor processes.
* The coordinate meta-analysis engines (``neurosynth``, ``neurostore``)
  have the peaks in MNI millimetres, ``peak_mni(x, y, z, study)``, the
  Destrieux atlas of the cortex and the Tian atlas of the subcortex beside
  Schaefer, and for each atlas ``{atlas}_activated(label, study)``: the
  studies with a peak within 3 mm of a voxel of a region.  The LLM
  assistant answers their questions with probabilistic query answering
  (an engine's ``assistant_guidance`` goes into its prompt), with
  Destrieux for the classical anatomy and Schaefer and Tian for the
  functional anatomy; the neurosynth example bank is rewritten so, with
  examples of the co-activation of regions and of points.

* The LLM assistant writes the query it accepted with a comment line
  explaining each rule, from one more request to the model (``%`` lines
  in Datalog, which runs as it is; ``comment_rules: false`` turns it
  off).  ``.llm --steps N``, ``.ask --steps N``, ``.llm steps N`` and
  ``--steps N`` on the command line set the most steps of the ReAct
  agent.

* In the REPL, the rules of a submission with a query (``ans`` or
  ``obtain``) are local to it: they are discarded after it, so that two
  queries defining ``active`` do not join their rules.  ``.scope off``
  keeps them in the session, as before.  Comments are shown in a lighter
  grey.

* ``.view`` and ``.save`` (and ``--view``, ``--save``) paint a result on
  the regions of an atlas, ``ans(l, p)``, region by region: a sort of
  region labels names its atlas with ``atlas`` in ``engines.yaml``.
  ``.view`` shows a result of pairs of regions or points with a value,
  such as their co-activation, as a network drawn by NiiVue's connectome
  view, over the brain.

* The page of the REPL in the browser has the logo, a header with links to
  the documentation, the tutorial and the source, the command it runs to
  copy, and a status that shows when it is busy; the choice of the
  NeuroLang version is folded away.

* The REPL in the browser shows a result as a table to sort by clicking
  a column, to filter (``stroop`` keeps the rows with the word,
  ``name:stroop`` those whose column ``name`` has it), to page through, to
  copy and to download as CSV, instead of the text table of the terminal;
  ``Ctrl+R`` searches the history of the inputs, which lasts between
  visits, as in a shell (``Ctrl+R`` again for an older match, ``Ctrl+S``
  for a newer one, ``Enter`` to take it, ``Esc`` to cancel), and Up and
  Down browse it.  :class:`neurolang.utils.tui.web.WebRepl` returns a
  result's rows as data (``Reply.tables``).
  
* An LLM query assistant writes a Datalog or SQUALL query from a
  question in natural language: ``.ask QUESTION`` in the interactive
  mode, which puts the query in the next prompt (``.llm QUESTION`` is
  the shortcut for the ReAct agent writing Datalog), and
  ``neurolang-query --ask QUESTION`` on the command line, where ``--run``
  also runs the query and ``-v`` and ``-vv`` show the reasoning and tool
  calls of its steps, and the messages sent to the model.  Two
  strategies are offered: in-context learning, whose few-shot examples
  are the closest to the question (BM25), with validation and repair on
  the engine, and a ReAct agent that explores the engine with tools
  (``search_values``, ``run_query``, ``check_query``…), by native tool
  calls or a text protocol.  The providers, OpenRouter, Albert,
  Anthropic, any OpenAI- or Anthropic-compatible server or a local
  command, are set in a YAML file, their keys read from the environment
  (``neurolang/utils/llm/llm.example.yaml``).  The queries are shown on
  their lines, highlighted.  It needs the ``llm`` extra and is described
  in :doc:`query_assistant`.  The benchmark that measures it is in
  `neurolang-text2query-bench
  <https://github.com/NeuroLang/neurolang-text2query-bench>`_.
  
* ``neurolang-query --save PATH`` writes the result as CSV or TSV, or as
  a NIfTI image on the grid of the engine, and ``--view`` shows its image
  in the viewer of the REPL, as ``.save`` and ``.view`` do (they need
  the ``tui`` extra).

* The interactive query REPL runs in the browser, with nothing to
  install: https://neurolang.github.io/_static/repl/index.html runs
  ``neurolang-query -i --engine neurosynth --resolution 2`` on Pyodide,
  with the REPL's commands and output, and ``.view`` shows its viewer
  below the output.  ``?ref=BRANCH`` in its address runs the NeuroLang
  of that branch, tag or commit, read from GitHub (``&repo=OWNER/NAME``
  for a fork).  It is :class:`neurolang.utils.tui.web.WebRepl`
  driven by a page (:file:`doc/_static/repl/`), whose bundle, NeuroLang's
  wheel and the data of the engine, :file:`doc/make_repl_bundle.py`
  builds with the documentation.  To run in Pyodide, ``pysdd``, which has
  no WebAssembly build, became optional, as the queries that lifted
  inference solves do not need it, and queries run in the calling thread
  where threads cannot start.

* Engines in :file:`engines.yaml` can inherit from other engines with
  ``extends`` and from ``abstract`` base entries, so settings common to
  several engines are declared once.  ``neurosynth`` now extends a shared
  ``coordinate_meta_analysis`` base.  See :doc:`engine_configuration`.
  
* A new ``neurostore`` engine, built on the same base, loads the NeuroStore
  database (as fetched by NiMARE) with the predicates of ``neurosynth``,
  TF-IDF computed from the abstracts (words and sequences of up to three
  words), the titles, abstracts and LLM-extracted concepts and domains of
  the studies, and SPECTER2 ``query_similarity`` over its abstracts
  (a ``text_embeddings`` source can now be engine predicates,
  ``source: {predicate: ...}``).  A ``brain_mask(i, j, k)`` predicate gives
  the whole brain (MNI152 brain mask) next to ``cortical_mask``.
* The ``neurosynth`` and ``neurostore`` engines define the lift, relative
  risk and uniformity ratio of a term, Neurosynth's association and
  uniformity tests, as predicates: ``uniformity('working memory', i, j, k, u)``,
  ``lift``, ``relative_risk`` and ``ratios`` (over the whole brain, with a new
  ``selected_voxel`` choice); see :ref:`tutorial-neurosynth-tests`.
  With SPECTER2, ``uniformity_llm('a description', i, j, k, u)``, ``lift_llm``,
  ``relative_risk_llm`` and ``ratios_llm`` do the same for a free text: the
  studies of the term are the 5% closest to it (``study_similarity``,
  ``similar_study``).
* ``neurosynth`` and ``neurostore`` load the Cognitive Atlas ontology
  (``ontologies``, now loaded before ``datalog_init``) for open-world queries
  over its concepts, tasks and contrasts, joined with the studies through
  ``cogat_subclass_of``, ``cogat_measures`` and ``cogat_term``; see
  :doc:`tutorial_cognitive_atlas`.
* NeuroLang has a logo (:file:`doc/_static/logo/`): the brain written as
  a probabilistic program. It heads the README and the documentation,
  whose colours become those of the logo, and is the favicon of the
  documentation, of the web application and of the ``.view`` viewer.

* The ``.view`` viewer has the controls of NiiVue's display: the view
  (a slice, the three slices in a row, a column or a grid, with or
  without the 3D render, or the 3D render alone), the drag mode of the
  right button (contrast, pan and zoom, measure, angle, 3D slicer), the
  crosshair's colour, width and 3D drawing, the gamma of the background,
  NiiVue's colour bars, labels, orientation cube and ruler, the
  radiological convention, the interpolation and a clip plane; the hot
  keys V and C keep the controls in step, and Reset restores them.
* Documentation: the REPL example of the Neurosynth tests
  (:ref:`tutorial-neurosynth-tests`) gives the term in the query,
  ``ratios("vision", i, j, k, ...)``, rather than in a
  ``term_of_interest`` relation joined in every rule, with a ``%``
  comment before each line.

* The ``.view`` viewer draws its slices with NiiVue (0.69.0, inlined in
  the page, which still needs no network but a browser with WebGL2): the
  order its overlays are drawn in can be changed with ▲ and ▼, and the
  mouse wheel scrolls through the slices.

* Documentation: Neurosynth's association and uniformity tests written as
  ratios of probabilities (lift, relative risk, and the ratio to the
  brain average), computed in the REPL of the ``neurosynth`` engine, see
  :ref:`tutorial-neurosynth-tests`.

* Documentation: the mutual information of a term and the activation of a
  voxel, a rule over the engine's ``p_term``, ``p_a`` and ``p_a_given_t``,
  ranks the voxels as Neurosynth's association test does, which the lift
  does not: for *prefrontal*, the Dice coefficient of its top 5% of the
  brain with that of Neurosynth's map is 0.83, against 0.51 for the lift.
  See :ref:`neurosynth-mutual-information`.

* ``.view`` of a result with several value columns shows an overlay per
  column, each with its own threshold, colormap, symmetry, opacity and
  visibility, a colour bar and its value at the cursor; ``--fill V`` sets
  the value of the voxels without one (default 0, hidden). An infinite
  value is drawn as the largest finite one.

* The ``neurosynth`` engine has a ``cognitive_task(task_id, name,
  description)`` relation: the tasks of the Cognitive Atlas, downloaded
  from its API (``cognitiveatlas.org/api/v-alpha/task``), with their
  descriptions, e.g.
  ``ans(d) :- cognitive_task(t, "stop signal task", d)``.

* The ``neurosynth`` engine has a ``cortical_mask(i, j, k)`` relation: the
  voxels of the MNI template grid in the cortex, labelled by the
  Harvard–Oxford cortical atlas (maximum probability, 25% threshold),
  resampled by nearest neighbour, for instance to restrict
  ``voxel_given_term(t, i, j, k, p)`` to the cortex.

* Text embeddings work offline: SPECTER2, its tokenizer and its adapters
  are downloaded once into ``{data_dir}/<engine>/embeddings/models/``
  (by the first query or by
  ``python -m neurolang.utils.text_embeddings``) and then loaded from
  there without network.

* The ``.view`` brain viewer of the interactive REPL starts with the
  ``plasma`` colormap and shows the top 5% of the voxels (threshold at the
  95th percentile of the nonzero values) unless ``--cmap`` or
  ``--threshold`` say otherwise.

* ``percentile(v, q)`` aggregation: the ``q``-th percentile of a column,
  ignoring NaN values, to keep for instance the top ``p``% of a column
  with ``v >= percentile(v, 100 - p)``.  The built-in aggregations
  (``count``, ``max``, ``mean``, ``sum``, ``std``, ``percentile``) are now
  also available in :class:`~neurolang.frontend.NeurolangDL`.

* Semantic search over document texts with SPECTER2
  (:mod:`neurolang.utils.text_embeddings`).  Engines declare a
  ``text_embeddings`` section in ``engines.yaml`` to register a function
  ``f(query, doc)`` returning the cosine similarity between a document and
  a free-text query.  The ``neurosynth`` engine provides
  ``query_similarity``, used as
  ``study(s) & (sim == query_similarity('visual word form reading', s))``,
  over PubMed titles and abstracts; see the new gallery example
  :ref:`sphx_glr_auto_examples_plot_text_embeddings_forward_inference.py`.
  Requires ``pip install "neurolang[embeddings]"``.

* Declarative engine registry for the ``neurolang-query`` CLI.
  Engines are defined in YAML and initialised by pluggable Python modules
  under :file:`neurolang/utils/engines/`.  Includes ``--list-engines`` for
  discovery and ``--list-predicates`` showing declarative metadata without
  data download.  (:mod:`neurolang.utils.engine_registry`,
  :mod:`neurolang.utils.engines`)

* The existing NeuroSynth and Destrieux engines were refactored into the
  new declarative system.  Engine init code moved from
  :mod:`neurolang.utils.cli` to :mod:`neurolang.utils.engines.neurosynth.init`
  and :mod:`neurolang.utils.engines.destrieux.init`.

* ``neurolang-query --list-predicates`` no longer requires downloading data;
  predicate metadata is read from the YAML engine config.

* **Datalog init** — engines can declare inline Datalog rules that run after
  the Python init via the ``datalog_init`` YAML field.  Derived predicates can
  be defined without writing Python.

* **CSV/TSV relations** — engines can load tabular data files as extensional
  predicates with the ``relations`` YAML field.  Supported formats:
  ``.csv``, ``.tsv``, ``.csv.gz``, ``.tsv.gz``.
  Relation entries may include an optional ``description`` field for
  predicate metadata shown by ``--list-predicates``.  The ``file`` value
  may be a URL for automatic download.

* **Builtins** — engines can declare ``builtins: [exp, log, startswith]``
  in YAML to register known functions as callable symbols without Python.

* **Downloads** — the ``downloads`` section fetches files from URLs
  (with optional archive extraction) before other init phases.

* **Probabilistic choices** — engines can declare uniform probabilistic
  choices declaratively with ``probabilistic_choice:`` instead of calling
  ``add_uniform_probabilistic_choice_over_set`` in Python.

* **Ontology loading** — the ``ontologies`` section loads OWL/RDF
  ontologies from URLs or local paths.

* **Atlases** — engines can declare ``atlases:`` with nilearn atlas names
  (``destrieux``, ``schaefer``, ``difumo``) and parameters.  Atlas regions
  are loaded as ``ExplicitVBR`` predicates before Datalog init rules.

* **Declarative migration** — ``exp``, ``log``, ``startswith`` symbols
  now come from YAML ``builtins`` rather than ``base.py``; the
  ``selected_study`` choice for the neurosynth engine is now declared
  in the engine YAML instead of Python init code; the ``destrieux``
  predicate is now declared in the YAML ``atlases:`` section.

* **Base symbols** — engines can set ``use_base_symbols: true`` in YAML
  to register common neuroimaging symbols (``agg_count``,
  ``agg_create_region``, ``agg_create_region_overlay``,
  ``principal_direction``, ``region_union``) automatically, without a
  Python init script.  ``region_union`` was moved into
  :func:`~neurolang.utils.engines.base.init_base_engine` as part of this
  change.

* **YAML-only engines** — the ``destrieux`` engine no longer needs a
  ``python_init`` module.  It is now fully declarative, using
  ``use_base_symbols`` and the ``atlases:`` section.  The
  :file:`neurolang/utils/engines/destrieux/init.py` module was removed.

* NeuroLang is layered as the packages it is to be split into: the
  expression engine (expressions, walkers, types, logic, config), the core
  (relational algebra and its sets, Datalog, probabilistic), the language
  (parsers, frontends, regions) and the applications (command line,
  terminal interface, engines).  Each layer imports only those below it,
  which ``.importlinter`` checks in CI: ``import neurolang.expressions``
  needs neither pandas nor the relational algebra sets, and the core
  needs neither the frontends nor lark, nibabel or nilearn.
  ``import neurolang`` imports the frontends when one is first used.
  The FOL to Datalog translation moved from ``neurolang.logic.horn_clauses``
  to :mod:`neurolang.datalog.horn_clauses`, and the Datalog pretty-printer
  to :mod:`neurolang.datalog.pretty_printer`
  (``neurolang.frontend.datalog.pretty_printer`` still imports it).
  Modules that choose the classes of the relational algebra backend when
  imported register with ``config.reload_on_backend_switch``, rather than
  being listed in the configuration.  The tests of the core that ran
  queries through a frontend moved to :file:`neurolang/frontend/tests`.

**Bug fixes:**

* The REPL block of :ref:`tutorial-neurosynth-tests` names the rules
  restricted to the cortex with ``_cortex`` (``ratios_cortex``,
  ``selected_voxel_cortex``, …): pasted in the ``neurosynth`` engine,
  which now predefines ``selected_voxel``, ``p_a``, ``p_a_given_t`` and
  ``ratios`` over the brain, it failed on its second definition of
  ``selected_voxel`` and added its rules to those of the others.
  
* The badges of the README: the coverage badge, deployed to ``gh-pages``,
  was deleted by each deployment of the benchmarks' dashboard there (and
  deleted it in turn): it has its own branch, ``badges``. The Python badge
  reads the versions from ``pyproject.toml``, as the release on PyPI
  declares none, and the static "code style: flake8" badge, which nothing
  checked, gives way to those of the documentation and spelling workflows.

* A Euclidean distance bound to a point given by numbers,
  ``peak_mni(x, y, z, s), (euclidean(x, y, z, -24, -4, -20) <= 3)``, was
  computed for every row (14 s on the peaks of neurosynth), and the
  reserved ``EUCLIDEAN`` raised an error: the point is now ranged by a
  stored relation of its own, so the points within the distance are
  precomputed as for two relations (well under a second).

* A probabilistic query with a variable occurring only in deterministic
  relations, ``q(x, PROB) :- r(x, y) & selected(x)``, gave the sum of the
  probabilities over the values of ``y`` rather than their probability,
  e.g. 1.0 rather than 0.5 for an ``x`` of two rows of ``r``: those rows
  are the same event.  The lifted plan of the deterministic subquery
  ``∃y r(x, y)`` keeps its tuples with a probability of 1, and the small
  dichotomy theorem solver leaves such a query to the lifted solver.
  
* A probabilistic query no longer fails with ``AttributeError: type object
  'Unknown' has no attribute '__origin__'`` when the program has a
  query-based probabilistic choice, such as ``selected_voxel(i, j, k) ^ (1 /
  m) :- ...``, that the query does not use, or whose rule's body is empty:
  the choice is then over no tuple, as a query-based probabilistic fact
  over no tuple already was.

* The REPL skips the ``%`` comment lines between statements, e.g. before
  a dot-command of a pasted block or in a piped batch, which it submitted
  as programs that failed to parse; a comment line within a statement of
  several lines no longer ends it.

* Magic sets leave out of a magic rule the atoms sharing no variable with
  its bound arguments, whose product with the magic set was only
  projected out: the Neurosynth tests of :ref:`tutorial-neurosynth-tests`
  with the term in the query, ``ratios("vision", i, j, k, ...)``, take
  13 s instead of 63 s.

* Magic sets pass the values of a stored relation with a constant to the
  rules after it in a query: ``ans(i, j, k, p) :- cognitive_task(_,
  "color-word stroop task", d) & voxel_given_sim(d, i, j, k, p)`` gives
  the description of the task to the rules computing the similarities to
  it, which raised an ``UnboundVariablesError``. A recursive component of
  a program is now a stratum of its own, before the rules depending on
  it, such as an aggregation, which raised a
  ``NoValidChaseClassForStratumException``.

* Magic sets missed the initial magic fact of a constant when the same
  predicate, with the same adornment, was called with another constant in
  a rule rewritten earlier, e.g. ``a(y) :- r(1, y)`` and
  ``b(y) :- r(2, y)``, so one of the calls gave no rows.

* A rule variable defined by an equality over head variables, e.g.
  ``sim`` in ``close(q, s) :- study(s) & (sim == f(q, s)) & (sim > .75)``,
  was rejected as unbound when the rule was defined, although a query
  constant, e.g. ``ans(s) :- close('math', s)``, binds ``q`` through magic
  sets.

* The magic sets SIPS passed bindings from the predicates computed from
  within-language probabilistic queries, e.g. a conditional (MARG, ``//``)
  rule ``p``, which are solved after the magic predicates:
  ``both(i, a, b) :- p('math', i, a) & p('lang', i, b)`` failed.

**Removed:**

* The legacy web application, ``neurolang/utils/server`` (2.3 MB, with
  ``papaya.js``), and the hatch build hook that built its frontend with
  npm, ``hatch_build.py``; the wheel no longer ships it.
* ``siibra`` and ``seaborn`` are no longer dependencies: NeuroLang never
  imports them.  The example
  :ref:`sphx_glr_auto_examples_plot_squall_bayes_factor_decoding.py` needs
  ``pip install siibra``.
* Unreferenced files: ``unused/neurosynth/cogat.v2.owl`` (1.4 MB), the
  pictures of an old Datalog grammar in
  ``neurolang/frontend/datalog/doc/`` (0.7 MB) and ``MANIFEST.in``.


v0.0.1 (Alpha)
--------------

Initial alpha release of NeuroLang.

**New features:**

* Datalog-based logic programming frontend (:class:`~neurolang.frontend.deterministic_frontend.NeurolangDL`)
* Probabilistic extensions with independent probabilistic facts
  (:class:`~neurolang.frontend.NeurolangPDL`)
* Tuple-generating dependencies (TGDs) for open-world reasoning
* Aggregation support (``COUNT``, ``MAX``, ``SUM``, custom callables)
* Neuroimaging integration: nibabel image loading, Destrieux atlas, NeuroSynth
* Example gallery: Destrieux gyri, sulcal queries, NeuroSynth implementation


How to add a changelog entry
-----------------------------

Add entries under **Unreleased** in the following format:

.. code-block:: rst

   Unreleased
   ----------

   **New features:**

   * Short description of the feature (`:class:` or `:func:` link if applicable)

   **Bug fixes:**

   * Short description of the bug and fix

   **Breaking changes:**

   * What changed and how to migrate
