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

- ``PROB[COUNT(x) >= k | a(g, x)] = p``: the probability that an event
  holds for at least (or ``>``, ``==``, ``!=``, ``<``, ``<=``) ``k``
  values of ``x``, per group; ``//`` conditions as in ``PROB``. The lifted
  solvers compute the tail of the Poisson-binomial distribution of the
  values in time linear in the data for any ``k`` (the plan of at least
  ``k`` values was ``O(n^k)`` for ``k`` at most 4, of a single atom); it
  now covers events of several atoms, deterministic ones and comparisons.

- ``EXPECT[f | a] = e`` and ``EXPECT_NORM[f | a] = e``, the expected value
  of ``f`` summed over the answers of a probabilistic query,
  :math:`\sum_x f(g, x)\,P[a(g, x)]` per group :math:`g` (``//`` conditions
  as in ``PROB``; ``EXPECT_NORM`` divides by :math:`\sum_x P[a(g, x)]`). It
  is sugar for a ``PROB`` rule and an aggregation, solved by the lifted
  solvers; the query assistant's bank has two examples of it, and
  ``NeurolangPDL.expect(f, event, result)`` builds it from Python. A rule that
  calls a relation inside an expression now fails with an error that says so.

- Query assistant: the provider ``api: claude-code`` has Claude Code run the
  ReAct agent (extra ``claude``): one conversation per question, native
  tool calls on the engine's tools, instead of a ``claude -p`` process per
  step, with your own sign-in to ``claude``, no API key. It is refused by
  ``neurolang-repl-serve --users``, unless an admin allows it to a user or
  a group (``--claude-code``, or the admin console). The ``claude-code`` entry of
  ``llm.example.yaml`` (``api: command``, ``claude -p``) is deprecated in
  its favour, as the entry ``claude-agent``.

- Brain viewer: the streamlines of a result with one other column, such
  as ``ans(x, side) :- uncinate(x, side)``, can be coloured by bundle, a
  colour per value of the column, which a legend names, or by direction,
  as before; *Colour by* chooses, starting at bundle. The column has 20
  values at most (``streamlines.max_bundles`` of ``viewer.yaml``); a
  streamline of several bundles is of the first, in alphabetical order.

- Brain viewer: the views take the width of the page, the 3 slices in a
  row as tall as a third of it allows (340 to 640 px), where they were
  960 by 340 px whatever the page; the options of the display start
  folded under *Display*, the image close to the threshold; the colour
  and the width of the crosshair are named for screen readers.

- ``neurolang-query --mcp`` and the REPL server's ``/mcp``: an answer's
  rows are shown by their value, highest first; its glass brain is a file
  too, linked; a page lists the answers of the session with their glass
  brains (``index.html``); the answers are kept (``answer.json``), and
  ``list_answers`` lists those of the earlier sessions, which ``ask``
  refines by their ids. New tools: ``examples`` (the assistant's example
  questions by topic) and ``status`` (the engine, its grid, its folders,
  whether it is built). ``--allow-dir DIR`` lets the confined statements
  read a folder of one's own (``.load`` reads the files of the folders
  they may read); the options may be kept in
  ``~/.config/neurolang/mcp.yaml``; ``--print-mcp-command`` prints the
  command adding the server to Claude Code; the data directory is
  ``~/neurolang_data`` by default. On the REPL server, each user's answers
  are in a folder of their own. The meta-analysis engines answer "where in
  the brain is X reported" with the lift of each voxel of the cortex.
  
- REPL server (``--users --local``): ``--session-memory-mb`` stops, on Linux,
  a query taking its session's process more than that much memory of its own,
  with a message, the session staying, instead of the OOM killer taking the
  builder or the server; ``doc/repl_server.rst`` gives the matching systemd
  ``MemoryHigh``/``MemoryMax``.

- REPL server (``--users --local``): it answers at once and prepares the
  engines in the background (those the page offers first), instead of
  after their preparation, which took minutes after an update that
  changed the tables kept (the page did not answer meanwhile). A session
  of an engine not yet prepared waits for it, its page telling so; the
  others start at once. Without ``--users`` the engines are still
  prepared before serving.
  
- REPL page: the question and query boxes have a copy button at their top
  right, and keep their own history of edits: Ctrl or Cmd Z undoes a run of
  typing at a time (the browser's undid a character at a time in Chrome) and
  also what the page wrote in the box (an example, the History menu, the
  assistant's query, Format); Shift Ctrl or Cmd Z, and Ctrl Y, redo.
  
- NeuroStore engine: a build no longer depends on the network when it has
  what it needs.  The listing of the releases is kept in the data directory
  for ``release_listing_ttl_hours`` (24, in ``engines.yaml``), a downloaded
  release is used as it is, a nightly release is downloaded again (and the
  old copy replaced only once the new one is there) when a fresh listing
  says it was rebuilt, and without the network the newest downloaded
  release is used with a logged message.  ``NEUROLANG_NEUROSTORE_UPDATE=1``
  lists the releases again.

- Magic sets no longer copy the rows of a stored relation they restrict:
  a restriction rule, ``p^bf(x, y) :- magic_p^bf(x), r(x, y)`` of a
  stored relation ``r``, is unfolded into the rules using it
  (``unfold_restrictions``, the last step of ``magic_rewrite``), whose
  plans join ``r`` with the magic predicate: relational algebra as any
  join, the contractions as a product with the sparse view of ``r``.
  Magic sets bind every predicate again, those of sparse views included
  (no ``unbound_predicates``); the decoding of a mask reads
  ``activated`` as its view, without the 20-million-row copy. See
  :doc:`probabilistic_lifting`, *Restrictions of magic sets left to the
  plans*.

- Probabilistic queries whose plans are sums of products (joins,
  projections and selections over choices, probabilistic facts and
  deterministic relations, the independent and disjoint projections of
  lifted plans, and conditionals) are computed as contractions of sparse
  tensors: the provenance plan is translated into tensor expressions
  (``nl_core.relational_algebra.linear_algebra.expressions``), rewritten
  into a normal form, and computed with ``scipy.sparse``, reading the
  engines' sparse views (``activated``, the atlas activations) as
  matrices. Plans it cannot write so are computed as before. With it,
  the restrictions of stored relations by magic sets are joined by the
  plans rather than copied (see below). Answers are
  unchanged (see :doc:`probabilistic_lifting`, *Computing the plans as
  tensor contractions*); ``[PROBABILISTIC_SOLVER] contractions = False``
  in ``config.ini`` turns it off.
- ``nl_core.relational_algebra.linear_algebra``, the encoding of
  relations for linear algebra: ``EntityIndex`` numbers the entities of a
  sort (a voxel ``(i, j, k)`` is one; strings through a dictionary) and
  back, and ``SparseView`` holds a relation between two entities, with
  the value of a third column or not, as a ``scipy.sparse`` CSR matrix
  with int32 indices, under set semantics, built once for the frame of
  its rows (``sparse_view``). An engine declares the views it builds with
  it, before the REPL server forks its sessions, which share them, in
  ``sparse_views`` of ``engines.yaml`` (``activated: [voxel, study]``),
  the entities of a sort numbered alike in all; the views of predicates
  the engine does not define (left out of a lightweight engine) are not
  built. The engines of the registry declare none yet, as no query reads
  them. ``SparseView.weighted()`` gives the matrix with float64 entries
  over the same indices, for products of views, made once with the views
  and shared by the sessions. Reading the rows of an
  empty relation of no columns no longer fails.

- The sparse views of an engine are kept with the tables of its build
  (``derived_tables`` of the data directory, a ``.npy`` file per array,
  named by a checksum of the relations they are built from) and read
  memory-mapped by the next build: their arrays (13 bytes a tuple, 1.8
  GB for neurostore_iit) are pages of the files, shared by the
  processes, by the engines built from the same tables, and reclaimed by
  the system when memory is short, rather than private memory of each
  builder; a build no longer spends time building them (24 s for
  neurostore_iit). A table a build reads back from its file is named,
  for what is found from it, by the name of the file, without reading
  its rows again.

- REPL in the browser: a *Network* tab draws an answer whose rows are edges
  (two columns of the same sort, such as ``concept_1`` and ``concept_2``,
  and a numeric weight) as a network laid out by force, with a slider for
  the weakest edge shown; it starts with the strongest, two per node, and
  says which edges and nodes it leaves out; of more than 40 nodes it names
  the most linked, the others under the pointer. The size and the shade of
  the nodes show a measure chosen (links, rows in a column, such as the
  children of a parent, or the nodes under it in a hierarchy), the shade
  of the edges their weight; a hierarchy can be drawn as a radial tree.
  It zooms (Ctrl or ⌘ and the wheel, a pinch, buttons) and pans, naming
  more nodes the closer; a node found by name or clicked is focused, with
  its neighbours, and a panel lists them to go from one to the next.
  The limits are in ``settings.yaml`` (``network``).

- REPL server (``--users --local``): with ``--mcp``, it answers MCP
  clients, LLM agents such as Claude Code, at ``/mcp``, each user with a
  personal token of their account page (only its hash kept, in the users
  file), in a session of their own whose REPL answers in plain text: the
  tools of ``neurolang-query --mcp`` over the network. The login page tells
  how to add the server to a client; the files of the answers (viewer,
  images, rows, query, graph) are kept in ``--mcp-results-dir`` and served
  to the users logged in, the pages sandboxed. The server's builder and
  sessions draw with matplotlib without a window: on macOS its default
  backend aborted a forked session drawing a glass brain.

- ``neurolang-query --mcp`` serves a session of the interactive REPL to an
  MCP client, such as Claude Code, over stdin and stdout (extra ``mcp``).
  The client's model answers questions as the query assistant does, in
  Datalog: ``ask`` gives the assistant's prompt for a question, the
  assistant's tools find the predicates and values and run queries, and
  ``final_answer`` checks the answer as the assistant's, then returns its
  explanation, the query with a comment on each rule, its first rows, glass
  brains of its images and the links to its viewer, images, rows, query and
  graph (``--results-dir``). ``--build-on-first-call`` builds the engine at
  the first tool call, and ``--log FILE`` writes the build and each tool
  call, with its time and outcome, to a file. ``ask`` with ``refine``
  changes an answer, or the last query run, for a new question; ``repl``
  runs statements and dot-commands in the session. See *From an MCP
  client* in the LLM Query Assistant documentation.

- REPL server (``--users --local``): the builder builds the engines the
  page offers when the server starts, the default first, and an engine
  the admin console turns on when it does, so that no session waits for
  a build (up to 75 s for ``neurostore_iit``); a session then starts in
  under two seconds. ``--no-prebuild`` (``prebuild: false``) leaves each
  engine to its first session, as before. The builder also finds the
  names the notes on a query's constants look up, once, rather than each
  session at its first query: that of ``neurostore_iit`` takes 6 s
  instead of 18 s. After each query, once it has replied, a session frees
  what the query left (``gc.collect`` and, on Linux, ``malloc_trim``,
  13 ms): with the chase fix of #1335, a session that decoded an image
  keeps 90 MB of its own instead of 0.8 GB.
- Query assistant, meta-analysis engines (``neurosynth``, ``neurostore``):
  the regions a question lists (which regions, parcels or structures a
  term, a topic or a seed activates or co-activates) are answered with the
  lift of each, P(active | topic) / P(active), or with the Bayes factor
  when the question decodes, rather than with the probability of their
  activation; the guidance and the examples of the assistant say so. The
  Schaefer atlas of the questions is that of 400 parcels: the engines
  have ``schaefer400_activated`` and ``schaefer400_covered``, and the
  assistant uses them unless a question asks for 100 parcels
  (``schaefer_activated``). They take about 90 s to compute the first
  time, then are kept with the engine's other tables (8 MB).

- REPL: the box of the assistant starts with one of the assistant's example
  questions chosen at random: filled in when the assistant can answer, as the
  grey hint otherwise. "Surprise me" picks another. Picking an example, a question or a query, fills
  both boxes, the English one and the query. The Questions list of
  "Examples and relations" is followed by the rest of the assistant's bank,
  "More questions", listed under topics (decoding, encoding, functional and
  structural connectivity, atlases...; the ``topic:`` of an entry of the bank
  overrides the one the page derives) and narrowed by a filter. Two new
  decoding examples: the terms of a Schaefer parcel and of a Tian region.
  
* On an engine of tractography (``neurosynth_iit``…), an uploaded
  tractogram is queried with the rules of the bundles: ``.load_tractogram``
  reads it by its name, and *Index*, on its row of the uploaded files,
  indexes it in place of the tractogram indexed before
  (``.index_tracts(..., replace=1)``, which removes that one from the
  relations, with a note); one tractogram is indexed at a time.  The
  engine's own tractogram is listed first, never removed, its *Index*
  putting it back (:doc:`tutorial_decoding`).

**Performance:**

- Lifted inference: the separator variable of a query need not be in the
  atoms of a deterministic relation, as ``organ`` in ``te_brain(d1, g),
  organ(d1, 'brain'), te_blood(d2, g), organ(d2, 'blood')``, which was
  declined to weighted model counting because ``organ`` is in two atoms
  without ``g``. See doc/probabilistic_lifting.rst, "Deterministic atoms
  of a separator variable".
- Lifted inference: the atoms of a tuple-independent relation selecting,
  by the deterministic atoms in their scope, disjoint sets of its tuples
  are atoms of new relations of these tuples
  (``rename_disjoint_restrictions``), without the self-join: the union
  ``R(x), S(x, y), D1(y)`` or ``S(x, y), T(y), D2(y)``, with ``D1`` and
  ``D2`` deterministic and disjoint, is lifted. See
  doc/probabilistic_lifting.rst, "Tuples deterministic atoms select
  disjointly".
- Lifted inference: a conjunction implies ``v != c`` when a positive
  deterministic atom of it has ``v`` where none of its tuples has the
  constant ``c`` of an atom of a probabilistic relation at ``v``'s
  position; the comparison is added before shattering
  (``ExcludedConstants``), which then removes the self-join through
  ``c``. ``via_thalamus(l) :- wired('aHIP lh', t), thalamus(t), wired(t,
  l)`` on neurosynth, declined to weighted model counting (12.9 s), is
  lifted (1.4 s), with the same answer. See doc/probabilistic_lifting.rst,
  "Constants a deterministic atom excludes".
- ``similar_study`` and the other rules of ``query_similarity``: the
  embeddings of query texts are kept on disk, next to the documents', so
  that a new process reads them rather than embedding the texts again
  (0.4 s a text with the numpy encoder), and ``query_similarity`` and
  ``query_similarity_cutoff`` are computed on whole columns (marked
  ``columnwise``): one product of the documents' embeddings for each
  distinct text, not a Python call a row. The bank's uncinate query of
  neurosynth_iit, which compares the 21 definitions of the concepts under
  'memory' with the studies (301 791 rows), took 19.7 to 20.3 s in a new
  process, now 12.1 to 12.2 s.

- ``uses(s) :- ri_task(t), skos:definition(t, d), similar_study(d, s)``
  computes ``similar_study`` for the definitions of ``ri_task`` alone, as
  fast as with ``skos:definition`` in a rule of its own: magic sets put
  ``skos:definition``, which binds ``d``, ahead of ``ri_task``, and the
  similarity was computed for every definition of the ontology (more
  than 300 s on the engine neurosynth). The rules sharing the variables
  of such a stored relation now bind before it, and the literals needing
  them after it.
- The magic-sets rewrite of a query over a large program (the tracts of
  an engine) took most of its time: the relations of the extensional
  database were built again for each literal checked (15 000 times), and
  rules were ordered definitions first by reading every body on each pass.
  A symbol table now keeps the symbols of each type it was asked for until
  it or a scope enclosing it changes (``symbols_by_type``, so
  ``extensional_database`` and ``builtins``), and each body is read once. The Fig. 2b query of the bundles co-activated by
  language studies on neurosynth_iit took 240 to 267 s, now 118 to 120 s,
  with the same answer.
  
- The magic-sets rewrite of the Fig. 2b query (neurosynth_iit) took 18.5 s
  of its 70 s, now 6.5 s, the rewritten programs of the three engines'
  questions and examples unchanged. Rules are ordered definitions first
  in one reading of their bodies, not one comparison of thousands of
  symbols a pass; ``symbols_by_type`` returns the read-only mapping it
  keeps rather than a copy, and ``extensional_database`` filters it once
  while it is kept, not once a literal checked (15 000 times). When the
  seeded rewrite finds no constant to bind, it is not tried again
  unseeded, which only binds less (on master before this change, 7 more
  attempts for each of the 3 answers).

- The semi-naive chase no longer evaluates again a rule with several atoms
  of its stratum's predicates when none of them has new tuples: the rules
  of tract queries rewritten with magic sets, 90 % of whose evaluations
  found nothing. A chase step looks up in the instance the relations its
  plan reads, rather than copying them all, and drops the tuples already
  derived only when it found some. The chase of the Fig. 2b query of the
  bundles co-activated by language studies (neurosynth_iit) took 46 s in
  1 334 steps, now 30 s in 952.

- Magic sets no longer make a non-recursive program recursive. All the
  call sites of a predicate with the same bound arguments share one
  adorned and one magic predicate, which can close a cycle of predicates
  none of which is recursive (``traverses^bbb`` reads
  ``magic_traverses^bbb``, which reads ``traverses^bbb``, in the WMQL
  tract bundles): the program was then one recursive stratum, rule by
  rule until nothing changed. Such a query is now solved unrewritten,
  unless it has a predicate bound only through its head
  (``similar_study``), which needs the rewrite. The rewrite leaves unbound
  at most one non-recursive predicate whose binding would leave the
  program unstratified, then gives up; a rewrite given up for a cycle of
  its own is not tried again without seeding from the stored relations,
  since binding fewer predicates does not undo the cycle. On
  ``neurosynth_iit`` at 2 mm, the ``af`` bundle took 22.5 s, now 7.1 s,
  ``extreme_capsule`` 19.4 s, now 4.3 s, and Fig. 2b's program without
  its bootstrap 19.4 s, now 14.8 s, with the same answers.

- An engine's sparse view keeps, with it, the sums of its entries for each
  row and each column entity (``SparseView.sums``, kept with the build's
  tables). A product of the view by the same number for every entity of
  one sort, summed over it, is that number times the kept sums: the view
  is not read. ``lift('pain')`` on neurostore, whose probability of
  activation over a uniform choice of all 64 295 studies read the 1
  billion entries of ``activated`` (57 s from disk), took 66 to 68 s in
  a new process, now 7.5 to 8.2 s; the lifts agree to a relative
  7.7e-14. A build reads each relation's frame once.

- A choice over a few rows of an engine's view (the voxels of an image) by
  the view reads those rows alone, and a vector over a few of its columns
  reads them through the view's index by column entities, rather than the
  whole view: decoding an image by the terms with ``selected_voxel`` on
  neurostore, 10 voxels against ``activated``'s 1 billion entries, took 7
  to 18 s, now 4 to 15 s; the product itself 1.7 s warm, 55 s cold, now
  next to nothing. ``FEW_CELLS`` and ``INDEX_SHARE`` are in the
  configuration: ``[PROBABILISTIC_SOLVER] few_cells`` and ``index_share``.

- A natural join of a small relation (``[RAS] join_cut_rows``, 10 000 rows)
  with one ``join_cut_ratio`` (100) times larger first keeps the rows of
  the larger one whose values of the common columns are among the smaller
  one's, before dropping its duplicates and merging (pandas). The magic
  set of a term, one row, was merged with the 10 M rows of NeuroStore's
  TF-IDF: ``mentions(s, 'pain'), mentions(s, 'emotion')`` took 9.2 s, now
  2.9 s.

- Decoding an image by the terms (``p_image_given_term``) took 110 s on
  neurosynth, now 3.7 s, with the same answer. Lifting tests whether
  columns of a relation determine another: two of its first rows
  (``[PROBABILISTIC_SOLVER] function_sample_rows``, 10 000) with one key
  and two values now disprove it, rather than counting the distinct
  values of the whole relation (346 M rows of ``activated``, 30 to 45 s a
  test). And a sparse view weighted by probabilities on a few of its rows
  (the voxels of the image) is multiplied by their diagonal, which reads
  those rows alone, rather than copied and scaled entry by entry (25 s).
  
- A selection or a projection by functions that apply to whole columns
  (comparisons, arithmetic, and functions marked ``columnwise``, as
  ``euclidean``) is computed once on the columns rather than once a row:
  the studies with a peak within 3 mm of an MNI point,
  ``peak_mni(x, y, z, s), (euclidean(x, y, z, -24, -4, -20) <= 3)``, took
  15 s over the 508 000 peaks of Neurosynth, now 1.2 s. A backend other
  than pandas computes it a row at a time, as before.

- Contractions multiply an engine's view by an operand of a few rows (the
  studies of a term) as a sparse by dense product, one pass over the view
  in its own layout, and build their operands with indices of 32 bits, as
  the views: ``uniformity('pain', ...)`` on neurostore went from 89-96 s
  to 63-68 s, the product of the term's studies by ``activated`` from 12 s
  to 2.8 s.

- A view can be kept by its column entities as well
  (``sparse_view_indexes`` of ``engines.yaml``; ``activated`` in the
  coordinate-based engines): a product with a few of them (the studies of
  a term) reads their entries alone, not the whole view.
  ``uniformity('pain', ...)`` on neurostore: 63-68 s to 13 s; the index
  of ``activated`` is 3.9 GB, built once.

- Magic sets: in the query's own rule, a small stored relation (at most
  ``[MAGIC_SETS] seed_rows``, 100 tuples) binds its variables without a
  constant, and is put first. ``chosen(t), uniformity(t, i, j, k, u)``
  computed ``uniformity`` for all 3 228 terms before keeping the chosen
  ones (30 minutes without an answer on neurostore); it now computes it
  for the chosen terms alone.

- Magic sets: of a cycle with an aggregation and a negation that a binding
  would create, the predicate left unbound is chosen in a fixed order, the
  aggregation's first. It was the first of a set, so a query could be
  planned differently from one run to the next: a functional connectivity
  query took 5 or 17 s, its aggregation computed once or twice.

- Engine builds read the cached atlas activations as arrays instead of
  rebuilding a list of tuples, and ``engine_cache.digest`` hashes string
  arrays as fixed-width bytes instead of through ``repr``; the cache
  ``VERSION`` is 2, so caches kept by earlier versions are found again once.

**Bug fixes:**

- A query of the Python query builder whose body holds a formula under
  ``&``, ``nl.exists(...)`` or ``nl.expect(...)`` (``nl.query((e.x,), e.a[e.x]
  & nl.exists(e.y, e.b[e.x, e.y]))``) failed with "Not a Datalog function
  application", since the check of the variables of its head read the
  formula as the argument of ``and_``; the free variables of an application
  now include those of its logic-operator arguments, bound variables
  excluded. This also failed the tests of ``nl.expect`` on master.

- An aggregation (or any rule reading it) over a ``PROB`` query without
  answer failed with ``NoValidChaseClassForStratumException``: the empty
  result was not in the solution of the probabilistic stratum, and the
  rules reading it could not be stratified.

- A join or a left join (and, with pandas, a difference) on a variable that
  one relation holds as text and another as numbers (the identifier ``'1'``
  and the number 1) used to fail with the backend's own error, a bare
  ``ValueError`` of pandas ("You are trying to merge on str and int64
  columns") or a ``SchemaError`` of polars; it is now one
  ``IncompatibleJoinColumnsError`` that names the variable and says what to
  check. Numbers of different types are joined as before, a relation with no
  row is not compared (as pandas did), and the comparison ``(t == 1)`` of a
  text is still false.

- The joins of a conjunction of hundreds of atoms are no longer a tree as deep
  as they are many, which the walkers that optimise and solve a plan recurse
  on one level per join: the optimiser of the plans of a probabilistic query
  failed with a ``RecursionError`` at about 250 joined relations. The
  translation of a rule body to relational algebra, the greedy ordering of the
  joins of the plans and the independent join of the lifted plans join in
  chunks of 16 operands where the operands each share a column with the one
  before (``join_in_chunks``); fewer operands, and a chain whose operands share
  none with the one before, are joined as before.

- The joins of hundreds of atoms that share nothing with the atom before them
  are no longer as deep as the atoms are many either: a rule body of such
  atoms (independent atoms, a centre joined to leaves that share columns with
  it only) is one n-ary join of its runs of atoms, and the independent join of
  the plans of components sharing no column is a tree of chunks; both used to
  fail with a ``RecursionError`` at about 330 atoms and 100 components. The
  other unions and joins of plans of ``dalvi_suciu_lift`` are built the same
  way. The columns of an n-ary join, and every ``OrderedSet``, are built
  without comparing their elements one by one, which took time quadratic in
  the number of columns of a join of hundreds of relations (a join of 200
  atoms that took 5.3 s now takes 3.9 s).

- A predicate used with fewer arguments than the columns of its relation,
  ``ans(x) :- p(x)`` for a binary ``p``, was solved on its leading columns,
  while one used with more arguments was an error; it is an error too, on
  both frontends, for stored relations and for those defined by rules, with
  a message that names the predicate and the numbers (``"`p` has 2 columns
  but is used with 1 argument"``). A predicate defined by rules and used with
  fewer arguments in the query of ``NeurolangPDL`` failed with "Relations
  must have the same arity". An empty relation has no known arity and is not
  checked.

- Lifted inference: a query with a component that is not liftable and
  independent components, as ``R(x), S(x, y), T(y)`` with ``n`` atoms
  ``U_i(z_i)`` on other relations, took a time exponential in ``n`` to be
  declined (``n = 8``: 264 s), on the lifted solvers and before weighted
  model counting on the default ones: the lifting enumerated the subsets of
  the clauses of the query; so did atoms of the answer variable of the
  core, ``U_i(w, z_i)``. The components that do not need each other to
  bind their free variables are now lifted apart (``n = 10``: 0.6 s).
  Queries whose independent component has an answer variable of its own,
  as ``h(z) :- C(x, y), R(x), T(z)`` with the choice ``C`` in a union, are
  lifted where they were declined.

- The choice facts of one predicate, ``0.3 ^ choice('a'). 0.7 ^ choice('b').``
  (the example of the tutorial, 10.6), are the alternatives of one choice,
  as the ``::`` facts of a predicate are one relation; the second one used to
  fail with "Probabilistic choice already defined". Their probabilities must
  sum to 1.

- Probabilistic queries of hundreds of conjuncts, such as a ``PROB`` whose
  event has 250 comparisons, no longer fail with ``RecursionError`` when
  they are prepared: the distribution of a conjunction over a disjunction
  (and of a disjunction over a conjunction), the standardisation of the
  quantified variables and the selection built by the shattering recursed
  once per formula. They now loop, and the standardisation takes the free
  variables of each formula once, not those of the formulas left at each
  step (150 comparisons: 3.6 s to 1.3 s; 200 atoms of 200 relations on one
  variable: 28 s to 15 s). The selection of the shattering nests the
  conditions differently above 16 of them.

- Magic sets: a rule using an intensional predicate with a number of
  arguments other than its definition's, such as ``cnt(x)`` for
  ``cnt(x, count(y)) :- par(x, y)``, failed with an ``IndexError`` (or an
  ``AttributeError`` in the chase of the rewrite). The rewrite now declines
  it (``PredicateArityInMagicSetsError``, naming the predicate and both
  arities) and the query is solved unrewritten, as for any other program
  magic sets cannot rewrite.

- ``PROB[exists(y; s(x, y), t(y))] = p`` (an existential over a conjunction,
  several variables, a disjunction) was dropped by the parser, the rule
  becoming ``q(p) :- True``, and ``MARG[s(x, y)] = p`` of a single atom
  failed or, with a group, lost it. They give the probability of their
  event, which the tests check against the enumeration of the possible
  worlds; a ``PROB`` or ``MARG`` whose event the parser cannot classify is
  now an error, not skipped.

- Magic sets: a query of the WMQL bundles ``uf``, ``ilf``, ``af`` or
  ``ec`` on the Disconnectome engines fits in memory again. Since the
  rewrite gave up on cycles of its own (#1391), these bundles were solved
  unrewritten, which computes ``endpoint_anterior_of`` and
  ``posterior_of`` whole: the product of 600 k endpoints, or 300 k
  streamlines, by 285 term boxes. ``ans(x) :- uf(x, "left")`` went past
  12 GB on ``neurosynth_disconnectome7t``; ``slf_i`` and ``cst`` fit,
  but lost their magic sets too. A program reading a stored relation of
  more than ``[MAGIC_SETS] seed_rows`` tuples is now first rewritten with
  the bindings of the constants, the heads and the needed arguments
  alone, which rarely has a cycle of its own: ``uf(x, "left")`` takes 7 s
  and 0.4 GB, against 12.5 s and 0.65 GB before #1391; ``af`` 14 s
  against 50 s; ``slf_i`` 0.2 GB against 0.5 GB on master.
- A query whose head has a variable missing from its body, as
  ``ans(streamline) :- uf(x, "left")`` or ``ans(study) :- both(s)``,
  raises ``UnboundVariablesError`` naming the variables and those of the
  body (*Use in the head the variables of the body (x)*). On a predicate
  defined by rules of ``NeurolangPDL``, it ran without the check and
  failed with an internal ``KeyError`` of pandas, and the assistant's
  ``check_query``, a dry run, found it valid; elsewhere it raised *All
  variables on the consequent need to be on the antecedent*, which named
  neither the rule nor the variables, as a rule of the program still
  did.

- AI assistant: an answer given when the ReAct agent ran out of steps,
  the last query that gave rows, says so: its summary reads *Out of steps:
  no final answer after N steps*, as a warning on the web page and in
  yellow in the terminal, where it read as a final answer.

- A conditional probabilistic query reading the result of another, as
  ``p_top(l, p) :- PROB[top_active(l, s) // tom(s)] = p`` with ``top(l)``
  a threshold on ``p_l(l, p) :- PROB[active(l, s) // tom(s)] = p``,
  raises ``UnsupportedProgramError``, as the unconditional one does,
  instead of the internal error *...\_noprob is already in the IDB with
  different signature*: rewritten with magic sets, the numerator and the
  denominator of a conditional query were given one predicate of two
  arities. The error says what is unsupported, and its suggestion how to
  rewrite the program: filter on a probability after the probabilistic
  queries, or store the first answer as a relation.

- A predicate whose rule binds a variable only through its head, as
  ``q`` in ``similar_study(q, s)`` of the engine neurosynth, gets the
  values of a stored relation of the rules using it:
  ``uses(s) :- cogat_term(t, 'stroop task'), skos:definition(t, d),
  similar_study(d, s)`` raised ``UnboundVariablesError``, as did
  ``study_similarity``, ``p_similar``, ``p_a_given_similar``,
  ``lift_llm``, ``uniformity_llm`` and the other ``*_llm`` predicates.
  When magic sets cannot bind every predicate without leaving the
  program unstratified (two WMQL bundles of the engine neurosynth_iit),
  only those predicates are bound, where the rewrite was dropped.
  
- Tractography: ``.index_tracts(t, a, replace=1)`` also removes from the
  relations the atlases that indexed only the tractograms it removes. Their
  labels stayed in the box of a term (``term_box_*``), so that the
  directional rules of the bundles (``endpoint_anterior_of``...) of the
  tractogram indexed depended on them: HCP842 indexed by ``cvs_avg35``'s
  wmparc in place of the IIT tractogram indexed by its wmparc (the commands
  of ``neurosynth_iit``) selected 715 left uncinate streamlines instead of
  tract_querier's 724, the left amygdala's box reaching 1.5 mm forward in
  IIT's wmparc and 0.5 mm in ``cvs_avg35``'s. The atlas removed keeps its
  options (``extent``, ``term_mapping``) for when it indexes again.

- An aggregation over a body with a negated atom, as
  ``r(sum(x)) :- p(x), ~q(x)``, is no longer refused as not conjunctive:
  it gives the aggregation over an intermediate rule holding the negation.
  An aggregate whose negated atom depends on its own head is refused, as
  with a positive one.
  
- REPL in the browser: the output of a query is legible on its dark panel.
  Its colours were Rich's, made for a light terminal: an error was dark red
  on navy (a contrast of 1.6). They are now the page's, each with a
  contrast of at least 4.5 (WCAG AA), dim text included.
  
- REPL in the browser: *Map the studies* is offered for the columns of
  studies only. A map of voxels, ``ans(i, j, k, lift)``, offered a map of
  the studies in ``i``, ``j`` and ``k``, its voxel indices taken for
  studies; integer columns of an unknown sort count only when no column is
  of studies.

- REPL in the browser: a map of voxels, such as ``ans(i, j, k, lift)``,
  no longer gets a *Network* tab: its ``i`` and ``j``, of the one sort of
  a voxel, were taken for the two ends of an edge. A sort of several
  columns now ends no edge; the page is sent the position of each column
  in its sort.

- A probabilistic plan computed as contractions whose answer could not be
  built as a relation set failed the query: the building of the answer is
  within the fallback, which logs the error and computes the plan by
  relational algebra, as for any error of the contraction.

- With the Polars backend, a probabilistic query computed as contractions
  failed (``IndexError``): the named relation sets of Polars take the rows
  of a pandas frame, as those of pandas do, columns in the set's order;
  columns of values Polars does not hold natively are read row by row.

- REPL server (``--users --local``): each engine has a builder process of
  its own, started by a coordinator, rather than one builder for all: a
  session of an engine built starts at once while another engine builds,
  rather than after that build: on ``neurosynth_iit`` while
  ``neurostore_iit`` builds, in 1.4-1.6 s rather than 96-98 s. A builder
  that dies fails the starts of its engine only, is told in the log, and
  the next session of its engine builds it again, rather than no session
  starting until a restart. One engine still builds at a time, in the
  order asked. The coordinator and each builder's own interpreter take
  about 0.5 GB more memory than one builder did (resident and private,
  both engines built, on Linux).

- The relational-algebra chase no longer keeps every chase that ran: three
  of its rewritings were cached with ``lru_cache`` on the method, a cache
  of the class whose keys held the chase, its program and the relations
  of its query. A REPL session kept 0.6 GB more after each query decoding
  an image on ``neurosynth_iit``; the caches are now each chase's own
  (the same hits), and that session keeps 0.18 GB, however many queries.
- Example programs that did not work: the start-up example of the
  Neurosynth and NeuroStore engines joining the Cognitive Atlas subclasses
  of *working memory* with the mentions returned no row (none of its seven
  subclass labels is a term of Neurosynth), it now takes those of
  *attention* (574 studies); two assistant examples on structural
  connectivity defined a probabilistic choice named ``ends_in``, a
  predicate of the vocabulary of tractography, and failed with
  ``ForbiddenDisjunctionError`` (renamed ``ending_region``); and the
  assistant example of the left uncinate fasciculus, which compared the
  definitions of the 583 Cognitive Atlas concepts with the 14,371 studies
  and timed out, takes the 21 kinds of *memory*.


v0.1a23 (2026-11-15)
--------------------

**New features:**

* ``destrieux_name(label, name)`` and ``tian_name(label, name)``, in every
  engine of the MNI atlases (and ``destrieux_name`` in ``destrieux``): the full anatomical names of the 148
  Destrieux regions (Destrieux et al. 2010, Table 1) and of the 32 regions
  of Tian's subcortex, scale 2 (the atlas's nomenclature), with their
  hemisphere, as 'Superior frontal gyrus (F1), left' or 'Ventroanterior
  thalamus, left', to join with a result of their labels; and
  ``schaefer_name`` (100 parcels) and ``schaefer400_name`` (400), the network
  and the component of each Schaefer parcel ('Default A network, dorsal
  prefrontal cortex 1, left', from the glossary of the atlas's authors), and
  ``atlas_abbreviation(atlas, abbreviation, meaning)``, what each
  abbreviation of the labels of the three atlases stands for; with
  ``lcase`` and ``str_contains`` to find a region by its name or an
  abbreviation, and examples for the query assistant.
  ``tools/make_atlas_names.py`` makes the tables from their sources.

* A tractogram is decoded as an image is, where its streamlines end:
  ``.image`` and **Upload data** take a ``.trk`` or ``.tck`` in MNI mm and
  load the voxels of the brain within 3 mm (``image.endpoint_radius``,
  ``--radius MM``) of the ends of its streamlines, each of probability 1,
  and draw its streamlines in the brain viewer; the page offers the
  decoding query, and the query assistant has an example.  The page's
  decoding buttons now run the query they put in the box.  The left arcuate fasciculus of HCP842 decodes to sentence
  comprehension, reading and language (:doc:`tutorial_decoding`).

* ``rdf:type(x, C)`` makes ``x`` an instance of the class ``C``, as RDF
  says, for the types a program states with rules as for those of the
  files: ``rdf:type(e, t) :- our_experiment(e, n), cogat_term(t, n)``
  types our experiments as Cognitive Atlas tasks by their names, and the
  Atlas's axioms then apply to them.  In a question, ``rdf:type(c, k),
  cogat_term(k, 'response inhibition')`` names the class by its label,
  for an unnamed ``c`` too, in the query and in the program's rules.  Only
  the classes with instances get them, so the rewriting still drops the
  others.  The example (IV) of :doc:`ontologies` is rewritten on it, with
  no identifier of the Atlas in its queries: our experiments, then the
  Stroop studies of Neurosynth, in the closed and the open world, every
  query of it now ending.

* Upload a brain image to the REPL: the page's *Upload data* takes a NIfTI
  image of activations (``.nii``/``.nii.gz``), a binary mask or a z-map,
  kept with the session as the CSV files are, so a server reading none of
  its files loads it.  On a server (``--users --local``), the files a
  session's page uploads go to a folder of its own in ``--uploads-dir``
  (``uploads_dir`` of ``server.yaml``), removed when the session ends,
  stopped or unused for ``--idle-minutes``, or when its process died; their
  sizes are its ``--max-upload-mb`` (50) and ``--max-uploads-mb`` (200).
  The admin console shows them by session, with their sizes, ages and the
  free disk space, and erases a file, a session's or all.  ``.uploads`` lists, in a session, the files its page uploaded, their
  kind, size and age, the relations loaded from them and the room left.
  The page has *Upload data* in its toolbar, and lists the files uploaded
  in the session (each reply carries them).  ``.image`` describes the image
  to the query assistant (its file, a mask or a map of values, its voxels),
  replacing the description of an image loaded before under the same name
  (``document_predicate(..., replace=True)``, ``.document(p, "...",
  replace=True)``).  ``.document(p, "...")`` in the REPL says ``documented p`` (not
  "nothing defined"), and the page keeps it with the rules of the session,
  after *Stop* or in a saved session.  ``.uploads remove NAME``, the
  *Remove* of the page's list, deletes an uploaded file from the session
  (from its folder on a server), and the page forgets it: the relations
  loaded from it stay until ``.drop``.  The list has a *Download* and a *Remove* button for each file;
  *Remove* leaves the output as it was and offers an *Undo*
  (``.uploads restore NAME``, the file removed last).  ``.image PATH|UPLOADED [--name
  NAME]`` loads it as ``NAME(i, j, k, value)``, NAME the file's name by
  default (``motor_map`` for ``motor map.nii.gz``; an upload of another
  file of the same name gets another), on the grid of the engine
  (resampled, a mask to the nearest voxel; at most ``image.max_mb``
  uncompressed, ``viewer.yaml``), shows it in the brain viewer and tells
  the query assistant what it is; the page gives it back after *Stop* or
  *Open session*.

* Decoding a brain image (:doc:`tutorial_decoding`): ``.image`` loads, with
  the image, ``image_probability(i, j, k, p)``, the probability that each
  voxel is active: 1 for a mask, or a map of values from 3.1 taken as one
  (``.image --mask``, masked decoding); for a map of z-scores, which
  ``.image --zscore`` says it is, 1 - the local false discovery rate of its
  z-score (Efron 2004: an empirical null of the map's median and MAD, a
  kernel density; ``image.min_probability`` and ``image.kde_sample`` in
  ``viewer.yaml``), with no threshold.  It prints, on the meta-analysis
  engines, the query decoding the image (``image_decoding_query`` in
  ``engines.yaml``, a comment explaining each line), and the page asks,
  after an upload, whether it goes in the box, and, for a map of values,
  how to decode it (*Decode as z-scores*, *Decode as a mask*), in the
  brain or, *Only the cortex* (``.image --cortex``), in its cortex: a plain
  Datalog query over the
  probabilities, so any relation of voxels and probabilities is decoded
  the same way.  Each voxel is a probabilistic fact of its probability and
  a voxel of the brain is chosen at random (``selected_voxel``); its score
  is the lift of each term: the
  probability that a study of the term activates the chosen voxel, which
  is active, over the same for all the studies.
  The query assistant's bank and guidance decode an uploaded image, and
  have the overlap query too (each voxel a probabilistic fact, a study
  hitting the image if it activates one), for a mask or a small region.

* The admin console of ``neurolang-repl-serve`` has *Update and restart*:
  it asks whether origin/master (the upstream of the server's checkout)
  moved, lists its new commits, and, confirmed, fast-forwards the checkout,
  syncs the environment to the lock with the extras installed (``uv sync
  --frozen --inexact``) and restarts the server in place; the page reloads
  once the server is back. A checkout with changes of its own is not
  updated.
* A dense whole-brain tractogram for tract_querier's bundles, the engines
  ``neurosynth_disconnectome7t`` and ``neurostore_disconnectome7t``:
  Disconnectome7T's tractography of the Human Connectome Project's 7 T
  diffusion MRI (Thiebaut de Schotten et al. 2020), 300,000 streamlines of
  at least 30 mm of 20 subjects (44 MB, 8 times the IIT tractogram's
  38,000), mapped into the HCP's MNI152 space, the space of the atlases, by
  an affine fitted on their subjects' own segmentations, all within the
  brain (the 7 T data are distorted over the orbits and under the
  cerebellum), with the FreeSurfer segmentation of those subjects.
  ``tools/hcp_connectivity`` downloads an OSF project (``osf``), checks
  and fits the space of a tractogram (``check-space``, ``fit-space``) and
  makes the tractogram (``dense``).

* The page of the REPL shows the answers of a program with several queries
  (``ans[name]``) in a tab each, named after its query, with its number of
  rows.  ``.view`` (and the brain tab) draws all those it can and leaves
  out, telling why, those with nothing to draw, such as a table of terms
  and Bayes factors beside the regions of the same program: it failed as a
  whole.  Each answer is an overlay of the viewer, which a box by its
  colour bar shows or hides; the masks of several answers have each a
  colour of their own (``mask_colormaps`` in ``viewer.yaml``: Blues,
  Oranges, Greens...), and the colours of the atlas start off, which would
  hide them.
  
* The connectivity of the brain from the Human Connectome Project, two
  small engines that ``coordinate_meta_analysis`` (``neurosynth``,
  ``neurostore``) extends, each of two relations over the cortex and the
  subcortex of an atlas, Schaefer 400 + Tian and Destrieux + Tian:

  - ``structural_connectivity``:
    ``schaefer400_tian_structural_connectivity(l1, l2, n)`` and
    ``destrieux_tian_structural_connectivity(l1, l2, n)``, the mean number
    of streamlines joining two regions over 100 HCP subjects' own
    tractography (MRtrix3: multi-shell CSD, anatomically constrained
    iFOD2, 5 million streamlines, SIFT2; Schaefer's and Tian's atlases
    brought to each subject through the HCP's warp, Destrieux's parcels
    of the subject's own FreeSurfer segmentation);
  - ``functional_connectivity``:
    ``schaefer400_tian_functional_connectivity(l1, l2, r)`` and
    ``destrieux_tian_functional_connectivity(l1, l2, r)``, the correlation
    of two regions' resting-state fMRI over 964 HCP subjects (ICA-FIX,
    MSMAll; the tanh of the mean Fisher z).

  Made by ``tools/hcp_connectivity`` (``tractography``, ``functional``,
  ``combine``, ``engine-tables``; ``--task``/``--tasks`` or a Slurm array)
  and published, with their standard deviations, the SIFT2 weights and a
  per-subject QC, at https://gitlab.inria.fr/dwasserm/neurolang-hcp-connectivity
  (tag v2.4), which the engines download, checked by md5. The cortex of
  the structural connectivity agrees with the connectomes of Mansour et
  al. (2021; Pearson 0.92 of the logarithms).  The guidance and the
  examples of the query assistant tell the three connectivities apart
  (structural, resting-state functional and the meta-analytic
  co-activation of regions by the same studies), turn counts and
  correlations into probabilities (computed, probabilistic facts,
  choices), and join them with the activations of a term.
  Two builtins, declared by the connectivity and meta-analysis engines,
  let a query test a difference and correct many tests:
  ``t_pvalue(t, df)``, the two-sided p-value of Student's t (computed
  from ``mean``, ``std`` and ``sum``), and the aggregation
  ``fdr_threshold(p, alpha)``, the p-value at or under which a group's
  tests are significant at the false discovery rate alpha
  (Benjamini-Hochberg).  The column of the structural connectivity is
  ``streamline_count`` (dataset v2.4), the mean number of streamlines, not
  SIFT2 weights.
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
* The coordinate-based meta-analysis engines (``neurosynth``,
  ``neurostore``) have ``schaefer_covered``, ``destrieux_covered`` and
  ``tian_covered(label, study)``: probabilistic facts of how much a
  study's activations cover a region, the mean over the region's voxels of
  a Gaussian kernel of 10 mm FWHM around the study's nearest peak
  (truncated at 3σ; facts below 0.001 left out), found once and kept in
  the engine cache.  The assistant's guidance and examples use them for
  graded questions (how strongly, how much of a region) in place of the
  yes/no ``{atlas}_activated``.
* ``add_probabilistic_facts_from_tuples`` and
  ``add_probabilistic_choice_from_tuples`` take a ``DataFrame`` (its
  column names were taken for its rows' type, and they failed).
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
* ``.ask --refine QUESTION`` and ``.llm --refine QUESTION`` change the last
  query for a new or extended question, instead of starting afresh: the
  assistant is given the previous question, the previous query, even if it
  failed, and what happened to it (the error, the first rows or that there
  were none, the engine's notes). The browser REPL has a *Refine* button next
  to *Ask*. The agents take ``run(question, previous=Previous(question,
  query, outcome))``.
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

* An engine names the sources of its data, with their citations and the
  acknowledgements their terms ask for (``acknowledgements``, inherited
  from the engines it extends), and the REPL's ``.credits`` prints them:
  the MNI template, the atlases, Neurosynth and PubMed, NeuroStore, the
  Cognitive Atlas, SPECTER2, the tractography atlases, the IIT Human Brain
  Atlas and the HCP; a test checks every engine names some. The REPL page
  has a second page, *Credits* (``credits.html``), that lists them for
  each of its engines, with links to their DOIs and a button to copy them.

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

* Assistant: the two requests that check an answer (its translation back
  into English and the judgement of it) start with the same message, the
  engine and its predicates, which the Anthropic provider now caches as a
  block of its own, so that the second reads it from the cache instead of
  sending it again, as does each check of a retried answer.

* The greedy join ordering starts from the join of two relations with the
  fewest rows at most, and adds at each step the relation whose join with
  the joined ones has the fewest rows at most, rather than the smallest
  relation: a relation all of whose columns are joined, which cannot add
  rows, is joined first.  Decoding a whole map of z-scores with its voxels
  as probabilistic facts and one chosen at random, which ran out of memory
  (more than 24 GB) on neurosynth at 2 mm, takes 13.4 s and 6 GB for the
  motor map of nilearn: the image is joined with the activations and
  summed per study before the terms.

* The plans of the lifted probabilistic queries sum each side of a join
  over the columns the projection drops before joining it
  (``PushSumsThroughJoins``), the
  product of the probabilities distributing over their sum: the same
  probabilities, from smaller joins.  Decoding an image with each voxel a
  probabilistic fact of its probability and one chosen at random takes
  8.4 s instead of 13.5 s on neurosynth
  at 2 mm for a mask, the voxels summed out per study before the terms
  are joined; ``uniformity('pain', ...)`` 5.9 s instead of 6.4 s.  The
  other queries of the example banks take the same time.

* The pattern matching of the walkers, which every compilation of a
  program goes through, tries for an expression only the patterns its class
  does not rule out, from a list made once per walker class, and builds its
  log messages only when logging is on.  The matches are the same; an
  open-world query on the Cognitive Atlas that the rewriting by its
  constraints dominates takes 215 s instead of 330 s.

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

* ``.view`` draws the regions of a result of the connectivity relations:
  their labels, of the sorts ``schaefer400_tian_label`` and
  ``destrieux_tian_label``, are looked up in ``schaefer400`` or ``destrieux``,
  then in ``tian``; a result of their regions beside names and numbers
  failed with *The result has no voxel columns*, and a result of pairs of
  regions is now a network.
* The engine ``destrieux`` builds: the ``description`` of an atlas of an
  engine is no longer passed to the atlas's fetcher (``fetch_atlas_destrieux_2009``
  refused it), and its ``left_region`` and ``right_region`` hold the
  regions of each hemisphere (they selected labels starting with ``lh`` and
  ``rh``, which none does: Destrieux's start with ``L`` and ``R``).
* ``exists(t; ...)`` in the rule of a query, ``ans(s) :- study(s),
  exists(t; mentions(s, t))``, failed in the probabilistic frontend, which
  the engines use ("Not a Datalog function application", or no
  ``functor``), and so did ``~exists(...)`` there; in a rule of the program
  it worked.  The rule of a query now goes through the translation to
  Datalog, as the other rules do: its ``exists`` and ``~exists`` become
  rules, and a Datalog query is left as it is.
* macOS: the query assistant's checks of a query no longer crash (SIGSEGV)
  on a matrix product, such as a bootstrap interval or a similarity of
  texts. Apple's Accelerate, the BLAS of numpy there, cannot run in a
  forked child once the parent ran it with several threads:
  ``neurolang_apps`` keeps it to one thread (``VECLIB_MAXIMUM_THREADS=1``,
  unless set), and checks in-process where numpy loaded first.

* ``.view`` drew nothing of a region, a result of voxels without a value,
  or of an overlay of one value: its default threshold, the 95th
  percentile, is that value, and NiiVue draws nothing of a colour range of
  one value.  Such a range now starts a colour below it, and the voxels
  are drawn in the top colour of the colormap.

* The semi-naive chase missed tuples in the first iteration of a recursive
  stratum: a rule's recursive atom saw only the tuples found earlier in that
  iteration, not those of earlier strata.  With ``rr(y, x) :- rr(x, y)``,
  ``p(x, y) :- rr(x, y)`` and ``rr(x, y) :- p(x, y)``, ``p`` lacked the
  first tuple of ``rr``.  In the first iteration a recursive atom now sees
  its whole relation.

* The constraints of an ontology ignored the instances a rule derives:
  with ``c:K(s) :- two_back(s)`` and ``c:K ⊑ ∃c:measures.c:WM``,
  ``ans(s) :- c:measures(s, c)`` returned nothing, while a fact of ``c:K``
  gave its answer.  The code a query needs is now followed through the
  constraints, so the rules defining the predicates of their bodies are
  solved with the query (:doc:`ontologies`).

* The classes and properties of an ontology named by their IRI, those of a
  namespace the file declares no prefix for, as the Cognitive Atlas's,
  could not be written in Datalog: ``<iri>(x)`` is now an atom of the
  predicate the IRI names, in heads and bodies.  ``load_ontology`` and the
  ``ontologies`` of ``engines.yaml`` take ``prefixes``, from prefix to
  namespace, and the ``neurosynth`` and ``neurostore`` engines name the
  Cognitive Atlas's ``cogat:CAO_00587``, ``cogat:measures``.  Their
  ``cogat_subclass_of`` is now the closure of ``rdfs:subClassOf`` on
  classes, whose values are IRIs, as ``rdfs:label``'s: that of
  ``neurolang:subClassOf``, on the names of the classes, would join with
  nothing, and gave the CogPO and SKOS classes by name.

* The rewriting of a query over an ontology drops a rewriting as soon as
  one of its atoms, over a predicate without values, can never be
  rewritten, instead of exploring all the rewritings it gives: on the
  Cognitive Atlas, ``ans(s) :- cogat:measures(s, c), cogat:CAO_00587(c)``
  explores 224 rewritings instead of 26,600 and takes 24 s instead of 60 s
  (:doc:`ontologies`).  The W3C OWL test cases in the fragment NeuroLang
  reads are run as tests, the others listed with the reason they are
  skipped.

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

**Changes:**

* ``similar_study(q, s)``, the studies about a text, keeps the top 5% of
  the studies closest to it only if the mean similarity of its 20
  closest studies reaches the ``similarity_gate`` of the engine's
  ``text_embeddings`` entry (0.75 by default, 0.76 for NeuroStore), and
  none otherwise: a text that no study is about, as 'stock market crash',
  got its 719 studies.
  The gate and the 95th percentile are the function
  ``query_similarity_cutoff(q)``, ``inf`` when the gate fails, computed
  once per text, rather than the aggregation ``cutoff_similarity``,
  removed: the studies of a Cognitive Atlas definition in the journal
  Neuron per year take 1.3 s instead of 70 s.  :doc:`similarity_calibration`
  describes how the gate is calibrated against texts off the domain, and
  its results on both engines: the 95th percentile of the null is 0.750
  over NeuroSynth's 14,371 studies and 0.762 over NeuroStore's 64,920,
  where 0.75 would let 9% of the null through.

* The SQUALL tests are skipped by default, ahead of the deprecation of the
  SQUALL syntax: the ``squall`` pytest marker is on the tests of its
  parser, grammar, translation, completion and command-line flags, and
  ``pytest --run-squall`` runs them.

* The page of the REPL, after a design review: the border of a field (the
  boxes, the selects, the query) is seen at 3:1 on its card, ``--control``
  in ``tokens.css``, where it was at 1.3:1, and the placeholder of the
  light theme at 4.5:1; a focused field keeps an outline in forced-colours
  mode; the page has its ``h1`` and the examples their heading; on a touch
  screen the buttons, menus, links and selects are 44 px high; the API key
  is the page's main button while the assistant has none; the shortcuts say
  ⌘ on Apple's systems; and the third theme is called Computer. A second
  look, at the page as it is with a result, the assistant at work and its
  menus, which is kept as a way to render these states without the engine
  (``doc/tools/repl_page_states``), found and fixed that on a phone the
  menus (More, Docs, a table's Download) opened off the left of the screen,
  out of reach, and now open in the flow, or rightwards; that the log of the
  assistant was wider than its box, its first line cut; that the faint text
  (the row numbers) was at 4.3:1 where a row is hovered; that the chips DOI
  and PMC were 14 px high targets; that the previous and next page buttons,
  the filter, the Download menu and the folds of the logs were short of
  44 px on a touch screen; that a sorted column's arrow was read aloud; that the ticks of a
  plot of 18 million and some said 18.1M twice; and that the "#" of a
  table's head touched its border. The login page has the field border and
  the faint text of the REPL's tokens, and the account and admin pages
  read their notes and states at 4.5:1 in a dark system and have 44 px
  controls on a touch screen, as the activity page, which had them only on
  narrow ones. The menus close with Escape and with a click elsewhere; the
  result card says "Running…" while a query runs, telling that the answer
  comes as a table that the Brain tab can then show on the brain, and shows a failed
  statement as one line with an icon above its log, which wraps; the steps
  of the assistant are open while it works; a disabled menu item says why in
  text; and, on a phone, the legend of the empty result still draws the
  probability bars it explains.

* The account and admin pages of the REPL server (``neurolang-repl-serve
  --users``) look like its login and activity pages: the tokens of the
  REPL's page, light and dark with the REPL's choice of theme, its logo and
  heading, cards and fields with a border read at 3:1, a focus ring on
  every control, a mark with the colour of an error or a success, 44 px
  controls on a touch screen, and tables that scroll by themselves on a
  narrow one, with their column heads scoped and the empty ones named.  The
  activity page keeps its theme under the same key as the others, so that one
  choice serves the REPL and all its pages.
  
* The slider of the streamlines in the brain viewer sets their number, from
  one to all of those the page has, on a logarithmic scale (a linear one
  would put the counts worth seeing in its first notch), where it set their
  opacity.  The file ranks the streamlines (``rank``, which NiiVue reads as
  a value of each, ``dps``) so that the ``k`` drawn are an even sample of
  them, spread over all, not the first ``k`` of the file, and the page asks
  NiiVue to hide those under the rank of the ``k``-th: the count is exact,
  not a multiple of a stride, and the label says it.
  
* The graph of a query (the Graph tab of the REPL, ``.graph``) opens at a
  size its text can be read at, where it was shrunk to fit its frame (about
  4 px text in the REPL's result column, 3 px on a phone): from the data on
  its left, in a box as high as a frame can show, which pans by dragging the
  mouse, with a finger or with the arrow keys and zooms around the point
  under the pointer, ``+`` and ``-`` too; ``Fit`` and ``Actual size`` say
  which is on, and a hint says when there is more to pan to.  Its intro is a
  line, with the long text and the descriptions of the legend folded under
  "How to read this", and its legend a row of short chips.  A node of an
  ontology is a card with the three linked dots of a knowledge graph in a
  band at its left, where it was an ellipse with four whiskers.

* The graph of a query says what a random choice is once, on its node
  ("random choice · 3 rows, 1/3 each", and "chooses one of its rows, each
  with probability 1/3" when it is clicked), where each probability that
  used it repeated it, and "so p is the share of those choices where A
  holds", in a paragraph of its own.  A probability keeps one line on how it
  is computed, ``p = P(A), the share of the choices where A holds`` (``p =
  P(A, C) / P(C)`` given a condition), so that its card is shorter.


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

**Security:**

* ``neurolang-repl-serve --local`` confines the statements of its users:
  with users or a password, the commands that read a file (``.load_csv``,
  ``.load_atlas``, ``.import``, ``.load_tractogram``, ``.load``, ``.view``)
  read the data directory only and fetch no address, and those that write
  one (``.save``, ``.transcript``, ``index_tracts``) refuse, so that a user
  can no longer read the server's files nor write them (``.save`` to a file
  of the account it runs as). ``--unconfined`` lifts it. The rule is
  ``neurolang.confine``, set by the variable ``NEUROLANG_CONFINE``; an
  address is any ``scheme:`` (``file:/etc/passwd`` too), and building an
  engine runs ``confine.lifted()``.
* The REPL page uploads the CSV and TSV files of its users (**Upload
  data**), which ``.load NAME`` loads: a session keeps them in its memory
  (at most 50 MB a file, 200 MB in all), and the page sends them again
  before it gives the rules of a session back (``.load`` is kept in a
  session, to save and to replay). A confined server's ``.load`` loads them only, and no
  path. ``POST /api/upload?name=`` takes the bytes, of the type
  ``application/x-neurolang-upload``, which no form of another site sends.

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
