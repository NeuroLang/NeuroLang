.. _neurolang-query-assistant:

LLM Query Assistant
===================

The query assistant writes a NeuroLang query, in Datalog or SQUALL,
that answers a question asked in natural language. It uses a large
language model (LLM). In the interactive mode, ``.ask`` puts the query
in the prompt, where it can be read, edited and run:

.. code-block:: text

    nl(neurosynth:SQ)> .ask Which voxels are reported by studies about decision-making?
    Asking claude-code (sonnet, react)...
      I need to check how the term for decision-making is stored.
      → search_values(predicate='term', column='term', pattern='decision')
        Values of term.term matching 'decision': 'decision', 'decisions',
        'decision task', 'decision making', 'lexical decision'
      The term is stored as 'decision making' (with a space), so I'll use that
      constant and check the query.
      → check_query(query="obtain every Voxel (?i; ?j; ?k) that a Study that
        mentions 'decision making' reports.")
        The query is valid. Its Datalog translation: (+4 lines)
      → run_query(query="obtain every Voxel (?i; ?j; ?k) that a Study that
        mentions 'decision making' reports.")
        14926 rows; columns i, j, k (+11 lines)
      → final_answer(query="obtain every Voxel (?i; ?j; ?k) that ...", ...)
        Accepted.
    ╭───────────────────────────── SQUALL query ─────────────────────────────╮
    │ obtain every Voxel (?i; ?j; ?k) that a Study that mentions 'decision   │
    │ making' reports.                                                       │
    ╰────────────────────────────────────────────────────────────────────────╯
    4 LLM calls
    ... the 14926 rows of the result, as a query prints them ...
    The query is in the prompt: edit it to run it again, or press Enter to see its result again.
    nl(neurosynth:SQ)> obtain every Voxel (?i; ?j; ?k) that a Study that mentions 'decision making' reports.
    The assistant ran this query: its result, not run again.

The ReAct agent runs its final query to accept it (``run_query`` is how it
tries one, ``final_answer`` runs the one it gives), and keeps the whole result:
it is printed right after the query, in the terminal and in the page, and
Enter on the query as it is shows it again, without running the query a
second time. Any edit, or any other input first, runs the query as usual, and
so does a result larger than 5 million cells, or ``.scope off``, which
keeps a query's rules in the session. The ICL strategy does not run its query:
Enter runs it.

From the command line, ``--ask`` prints the query, and ``--run`` also
runs it. ``--save PATH`` and ``--view`` run it too, and save its result
as a table or a NIfTI image, or show its image in the viewer (see "Saving
and viewing the results" in the Datalog tutorial):

.. code-block:: bash

    neurolang-query --engine neurosynth --squall --ask "Which studies mention pain?" --run

While it works, a line per step is printed on the standard error: the
tool called and the start of its result. ``-v`` prints the steps in
full: the model's reasoning, the tool called with its arguments and the
first lines of its result. ``-vv`` prints the conversation with the
model instead: each request, with the messages added since the previous
one (the system prompt and the question first, then the model's last
reply and the result of its action), and each reply with its time and
tokens.

.. code-block:: text

    $ neurolang-query --engine neurosynth --ask "Which studies mention pain?" -v
      thought: The question asks for studies that mention the term 'pain'. I can use the `mentions` predicate which relates studies to terms they mention.
      → final_answer(query="ans(s) :- mentions(s, 'pain')", explanation="This query retrieves all studies that mention the term 'pain' using the mentions predicate.")
        Accepted.
    -- This query retrieves all studies that mention the term 'pain' using the mentions predicate.
    ans(s) :- mentions(s, 'pain')

The assistant needs the ``llm`` extra: ``pip install "neurolang[llm]"``.

Strategies
----------

Two strategies are offered. They follow the techniques of text-to-SQL
and text-to-SPARQL systems.

``icl`` (in-context learning)
    The prompt holds a reference of the language's syntax, the
    predicates of the engine and a few example questions with their
    queries. The predicates are given with their columns, their
    description in the engine configuration and their size (schema
    linking). A predicate documented in the program with a ``%%``
    block (see :ref:`datalog-doc-comments`) is given with its
    description and the names and sorts of its arguments,
    ``uncinate(x: streamline, h: hemisphere) — The uncinate
    fasciculus…``, which win over those of the configuration: the first
    paragraph of the description in the prompt and ``list_predicates``,
    all of it in ``describe_predicate``. The examples are those
    closest to the question, chosen by
    BM25 from a bank of checked examples, the ``assistant_examples`` of
    the engine's configuration, kept in
    ``packages/neurolang_apps/neurolang_apps/utils/engines/<engine>/examples.yaml``
    (see :ref:`neurolang-engine-registry`).
    An engine inherits the banks of the engines it ``extends``: the
    ``neurosynth`` and ``neurostore`` engines share that of
    ``coordinate_meta_analysis``, and ``neurostore/examples.yaml`` adds
    questions on its concepts, domains and abstracts. The guidance of the
    prompt, ``assistant_guidance``, is in ``<engine>/guidance.yaml``.

    The columns of an answer are named by the variables of its ``ans``:
    the prompt asks for names that say what they hold,
    ``ans(region, probability)`` rather than ``ans(l, p)``, and the
    examples are written so (a test checks it). When the answer is
    computed over entities the question names, such as the regions at the
    ends of a tract, these are given too, as an answer of their own: one
    named query ``ans[name](...)`` per kind of rows, as in the example of
    the ``neurosynth_iit`` bank on the cognitive tasks of the studies
    activating the regions at both ends of the left uncinate fasciculus,
    which answers ``ans[uf_end_regions](region_1, region_2, streamlines)``
    and ``ans[task_specificity](task, bayes_factor)``.

    The query the model writes is then validated on the engine. It is
    parsed, then dry-run; unknown predicates and wrong numbers of
    arguments are reported. When it fails, the formatted error is sent
    back to the model for a correction, up to ``max_repairs`` times.

``react`` (ReAct agent)
    The model reasons and calls tools:

    - ``list_predicates`` and ``describe_predicate``, which shows a few
      rows of a predicate;
    - ``search_values``, to write constants as the engine stores them,
      e.g. ``'decision making'`` for "decision-making" (entity
      resolution);
    - ``check_query``;
    - ``run_query``, which shows the number of rows and the first ones;
    - ``final_answer``, accepted only if the query passes the checks.

    The tools are called with the provider's native tool calls, or with a
    Thought / Action / Observation text protocol that any chat model can
    follow.

    The final answer is then verified by back-translation. In a request
    of its own, which does not show the question, the model says in
    English what the query returns, from the predicates, the query and
    the first rows of its answer. In another request, it judges whether
    that answers the question, as the engine's ``assistant_guidance``
    asks it to be answered. A ``no`` rejects the final answer: the
    agent gets the English reading and the reason as the observation of
    its ``final_answer``, and corrects its query. ``verify_rounds`` (1 by
    default, 0 to turn the check off) is the number of answers the check
    may reject; after them, the answer is kept with a warning. The REPL
    and ``neurolang-query --ask`` show the English reading of the query.

With ``samples`` above 1, either strategy answers that many times, at a
temperature of at least 0.7, and the answer whose rows most of them give
is kept (self-consistency). With ``fallback_provider: NAME``, a provider
of the file, that provider answers once instead when no result is given
by more than half of the answers, or when none is valid (a single sample
asks it only when its answer fails): e.g. three DeepSeek answers, and
Sonnet when they disagree. The fallback is a step of the answer, with
the query the vote kept (``voted``), and its requests count in the usage.

The rules of the queries the assistant checks or runs are defined in a
scope that is discarded afterwards. They never stay in the session.
When you run the query put in the prompt, its rules are local to it as
well: in the REPL, the rules of a submission with a query (``ans`` or
``obtain``) are discarded after it, so that the next query can define
rules of the same names (``active``, ``seed``…) without adding to
these. ``.scope off`` keeps them in the session; rules submitted
without a query are kept in any case.

Once a query is accepted, one more request asks the model for a line
explaining each of its rules, which is written above the rule as a
comment. The query itself is the one that was checked: the model only
writes the comments, and a reply without one line per rule leaves the
query without them. In Datalog the comments are ``%`` lines, which the
parser ignores, so the commented query is the one put in the prompt or
printed, and it runs as it is; SQUALL has no comments, so its rules are
shown with their comments and the query is kept without them.
``comment_rules: false`` in the configuration leaves out this request:

.. code-block:: text

    $ neurolang-query --engine neurosynth --steps 12 --ask "Which pairs of Schaefer regions are co-activated by studies about reading?"
    % Identifies Schaefer regions activated in selected studies.
    active(l, s) :- schaefer_activated(l, s), selected_study(s)
    % Finds pairs of regions activated within the same study.
    co_active(l1, l2, s) :- active(l1, s), active(l2, s)
    % Selects the studies closest to a text.
    close(q, s) :- similar_study(q, s), selected_study(s)
    % The probability of co-activation given a study close to the text.
    p_co_active_given_close(q, l1, l2, PROB) :- co_active(l1, l2, s) // close(q, s)
    % The pairs co-activated by the studies about reading, by its Cognitive Atlas definition.
    ans(l1, l2, p) :- (l1 < l2), cogat_definition('reading', d), p_co_active_given_close(d, l1, l2, p)

Questions given as a description or a cognitive concept
-------------------------------------------------------

On the ``neurosynth`` and ``neurostore`` engines, a topic given as a
description, or a cognitive process, task or concept, is matched to the
studies by the similarity of their title and abstract to a text
(:doc:`tutorial_text_embeddings`). ``similar_study(q, s)`` holds for the
5% of the studies closest to the text ``q``, ``study_title(s, t)`` gives
their titles, and ``cogat_definition(t, d)`` the definition ``d`` of a
Cognitive Atlas concept by its lower-case label ``t``.  A question naming
a cognitive concept ("studies about working memory", "the n-back task")
without saying mention, term or word is answered with the concept's
Cognitive Atlas definition when ``cogat_term`` has its label, else with
the question's own words; ``mentions(s, t)`` is kept for the questions
about mentions, terms or words, and for the topics that are no cognitive
concept (anatomy, disorders, populations, stimuli).  Both strategies
know this from the engines' examples and guidance:

.. code-block:: text

    $ neurolang-query --engine neurosynth --ask "Which studies are about the Cognitive Atlas concept response inhibition, with their titles?"
    ans(s, title) :- cogat_definition('response inhibition', d), similar_study(d, s), study_title(s, title)

The label must be the Cognitive Atlas one: ICL writes it from the
question, and finds nothing for a concept the atlas does not have (it has
*Emotional regulation task*, a task, but no *emotion regulation*), while
the ReAct agent can look it up with ``search_values`` over ``cogat_term``.

The definition is joined in the queried rule itself, as here, or passed
to a rule taking the text as its first argument (``close(q, s) :-
similar_study(q, s), selected_study(s)``): a text joined inside another
rule is not passed on, and the query fails with an
:class:`~neurolang.exceptions.UnboundVariablesError`.  The similarities
come from the engine's text encoder, SPECTER2 by default; ``similar_study``
keeps a share of the studies rather than a threshold, so the queries do
not depend on it.

Configuration
-------------

The providers are configured in a YAML file. It is searched for in this
order:

1. the path given by ``--llm-config``;
2. ``$NEUROLANG_LLM_CONFIG``;
3. ``~/.config/neurolang/llm.yaml``;
4. the example shipped in ``packages/neurolang_apps/neurolang_apps/utils/llm/llm.example.yaml``.

.. code-block:: yaml

    default_provider: albert
    providers:
      albert:                         # Albert, the French State's LLM API
        api: openai
        base_url: https://albert.api.etalab.gouv.fr/v1
        api_key_env: ALBERT_API_KEY
        model: gemma-4-31b-it
        tool_calling: text
        requests_per_minute: 24       # the key's rate limit
      openrouter:
        api: openai
        base_url: https://openrouter.ai/api/v1
        api_key_env: OPENROUTER_API_KEY
        model: anthropic/claude-sonnet-5.5
        tool_calling: native
      anthropic:
        api: anthropic
        base_url: https://api.anthropic.com/v1
        api_key_env: ANTHROPIC_API_KEY
        model: claude-sonnet-5-5
        tool_calling: native
      claude-code:                    # Claude Code with its own login
        api: command
        command: [claude, -p, --model, sonnet, --tools, "",
                  --no-session-persistence, --setting-sources, "",
                  --system-prompt, "{system}"]
        tool_calling: text
    assistant:
      strategy: react        # or icl
      language: auto         # the REPL's .mode, or squall, or datalog
      max_steps: 16
      max_repairs: 3
      few_shot_k: 6
      samples: 1
      # fallback_provider: anthropic
      temperature: 0.0
      timeout: 60
      preview_rows: 10
      verify_rounds: 1       # ReAct: answers back-translation may reject
      comment_rules: true    # a comment line on each rule of the query

The ``api`` of a provider is one of:

``openai``
    The OpenAI chat completions API, spoken by OpenRouter, Albert, vLLM,
    Ollama and most open services. OpenRouter caches the prompts of
    DeepSeek and OpenAI models by itself, and those of Claude (``anthropic/``
    models) because the assistant asks it to.
``anthropic``
    The Anthropic messages API, with prompt caching: each request of the
    assistant repeats the system prompt and the conversation so far, which
    the API then reads from its cache, for five minutes, at a tenth of the
    price of input tokens or less. The line after the query counts the
    tokens read from the cache (``5 LLM calls, 40213 tokens, 31876 read
    from the cache``).
``command``
    A local command that reads the conversation on its standard input
    and writes the reply on its standard output. ``{system}`` in its
    arguments is replaced by the system prompt.

API keys are only read from the environment variable that
``api_key_env`` names: the file rejects an inline key.

The example configuration also has ``albert-deepseek`` (DeepSeek V4
Flash on Albert), and ``openrouter-gemma`` and ``openrouter-deepseek``
(Gemma 4 31B and DeepSeek V4 Flash on OpenRouter, without Albert's daily
quotas). DeepSeek writes longer replies: its entries set
``max_tokens: 4096``.

In the REPL, ``.llm`` shows the configuration. ``.llm provider NAME``,
``.llm strategy icl|react`` and ``.llm steps N`` switch the provider, the
strategy and the most steps of the ReAct agent (``max_steps``), and
``.llm reload`` reads the file again. ``.ask --squall``, ``--datalog``,
``--icl``, ``--react`` and ``--steps N`` choose the language, the
strategy and the most steps for one question, as ``--steps N`` does on
the command line. Out of steps, the agent answers with the most detailed
query it ran that gave rows.

``.llm QUESTION`` is the shortcut for ``.ask --react --datalog QUESTION``:
it always writes Datalog with the ReAct agent, whatever ``.mode``, ``.llm
strategy`` and the ``strategy`` and ``language`` of the configuration, and
puts the query in the next prompt, switching the session to Datalog.
``.llm --steps N QUESTION`` gives it at most N steps. A question cannot
start with ``provider``, ``strategy``, ``steps`` or ``reload``, which
change the settings:

.. code-block:: text

    nl(neurosynth:DL)> .llm --steps 12 Which pairs of Schaefer regions are co-activated by studies about reading?

``.explain`` reads the last query that ran, whoever wrote it, in English: the
model is given the query, in its language, with the engine's predicates,
and answers with a sentence or two, printed after ``Reads as:``. It takes
no argument, and does not run the query again.

.. note::

   The question, the description of the engine's predicates, some of
   their rows and the results of the queries the agent runs are sent to
   the provider.

In the browser
--------------

The `REPL in the browser <_static/repl/index.html>`_ runs the assistant with
your own API key, which you give in the page: Anthropic, OpenAI, OpenRouter
or any server speaking the OpenAI API at the address you give (it must
accept calls from web pages). The key stays in the tab, or on the device if
you ask, and is sent to the provider and nowhere else. ``.ask`` and ``.llm``
work as in a terminal; the query they write is put in the input of the page.

The section *Assistant steps*, under the question, shows the work of the
assistant as it happens, one line per call to the model and per tool call of
the ReAct agent with the queries it tries and what they gave back, as the
terminal prints them; it is closed until you open it. A step appears when its
model reply arrives (replies are not streamed word by word), the tool call
before its result.

The code box colours the query as the terminal does, with the same
tokenizer (``query_highlight``), in Datalog and in SQUALL.

.. _assistant-feedback:

Rating and correcting the answers
---------------------------------

The assistant can share its questions and answers with a feedback server,
where people mark them correct, incorrect or unsure and correct their query.
The answers, their ratings and the corrected queries are exported to train
or distil models and to grow the banks of examples.

Sharing is off by default. Turn it on in ``llm.yaml``:

.. code-block:: yaml

    feedback:
      enabled: true
      url: https://feedback.example.org   # a neurolang-feedback-server
      user: Ada                           # optional: your name on the ratings

or by setting ``NEUROLANG_LLM_FEEDBACK_URL`` to the address of the server.
The first answer of a session says that the answers are shared, and with
whom. If the server cannot be reached, a warning is printed once and the
assistant works as before.

**What is sent.** For each answer of ``.ask``, ``.llm`` or
``neurolang-query --ask``, including those without a valid query or that
failed for an error of the provider: the question; the query, its explanation, and
whether it is valid or why not; the steps of the agent (thought, tool,
arguments, and the first 1000 characters of what the tool returned); the
rows the agent saw when it ran the final query; the engine's name; the
provider's name and model; the strategy; the time, the number of LLM calls
and the tokens; the version of NeuroLang; and the optional ``user``. API
keys, the provider's address and headers, and the paths of the
configuration and the data are never sent, and the home directory is
written ``~``. The rows come from the engine, so do not share answers on
private data.

**In the REPL.** After a shared answer, its id is printed. Then:

``.rate good|bad|unsure [COMMENT]``
    Rate the last answer: correct, incorrect or unsure.
``.rate ID good|bad|unsure [COMMENT]``
    Rate the answer ID.
``.refine``
    Put the last answer's query in the prompt. Correct it and run it with
    Enter.
``.refine submit [COMMENT]``
    Check the query you last ran on the engine, preview its rows, and share
    it as the corrected query of the last answer. It is rated incorrect.
    A query that fails the check is not shared.

``.help feedback`` recalls these commands. In the browser, the REPL shares an
answer only when its visitor rates it, in a Grist document (see below).

**In a Grist document.** Instead of a server, the answers can be rows of a
`Grist <https://www.getgrist.com>`_ document:

.. code-block:: yaml

    feedback:
      enabled: true
      url: https://grist.numerique.gouv.fr   # the Grist instance
      grist_doc: YOUR_DOC_ID                 # in the document's address
      api_key_env: GRIST_API_KEY             # the default
      user: Ada

The API key (in Grist, *Profile settings*, *API*) is read from the
environment variable ``api_key_env`` only; without it, a warning is printed
once and nothing is shared. The document needs a table ``Answers`` with the
columns ``time`` (date and time), ``question``, ``query``, ``valid``
(toggle), ``error``, ``explanation``, ``engine``, ``provider``, ``model``,
``strategy``, ``language``, ``rows``, ``steps`` (JSON text), ``seconds``,
``calls``, ``tokens``, ``user`` and ``client_version``, plus the review
columns ``verdict`` (choice: good, bad or unsure), ``comment`` and
``corrected_query``. Each answer is a row, its id the row's number. Flag the
rows in Grist by setting their ``verdict``, or with ``.rate`` (``good`` or
``bad`` set the ``verdict``) and ``.refine submit`` (``bad`` and the
``corrected_query``).

**Rating an answer in the browser.** After an answer of the assistant, and
after a query typed in the page, the page asks "Did this query do what you
wanted?", between the output and the result:
*Yes* sends it as ``good``, *No* asks what was wrong and sends it as ``bad``.
Nothing is sent unless it is rated. An answer is sent as the terminal shares
it; a typed query with its first rows or its error, and an empty
``question``. Either is one new row of the table ``Answers`` with its
``verdict`` and ``comment`` set. The page has no API key: Grist does not accept the
``Authorization`` header of another origin, and a key in a page is public.
The document is the one of ``feedback.json`` next to the page (``url`` and
``doc``; without a ``doc``, the page does not ask), and must let anyone add
rows, and do nothing else, in its access rules (*Access rules*, then the
table ``Answers``: for *Everyone*, create only, no read, update or delete),
so that a visitor can neither read nor change what others sent. Two rules
for the table, in this order: the condition ``user.Access == OWNER`` gives
all permissions, and a rule without a condition gives ``+C-RUD``. A visitor
who reads, changes or deletes a row is then answered "Blocked by table ...
access rules", and adding a row works.

**Running a server.** ``neurolang-feedback-server`` comes with
``neurolang-apps[llm]`` and adds no dependency: it runs on ``http.server`` and
SQLite.

.. code-block:: console

    $ export NEUROLANG_FEEDBACK_ADMIN_TOKEN=$(openssl rand -hex 16)
    $ neurolang-feedback-server --db feedback.sqlite --host 0.0.0.0 --port 8765

Its page at ``/`` lists the answers, filtered by status (``unrated`` or the
majority verdict) and engine. On the page you can rate an answer and correct
its query. A query corrected on the page is marked unchecked, because it was
not run on the engine. The API is:

* ``POST /api/answers``: post an answer;
* ``POST /api/answers/ID/feedback``: post a rating;
* ``GET /api/answers?status=&engine=&limit=&offset=``: list the answers;
* ``GET /api/export.jsonl``: export everything. It needs the header
  ``Authorization: Bearer TOKEN``, and is disabled when no token is set.

Anyone who can reach the server can post answers and ratings: there is no
login. Requests are limited in size (``--max-bytes``, 1 MB) and in rate per
address (``--rate``, 60 posts a minute). ``--cors-origin`` restricts the web
pages that may post. Put the server behind a reverse proxy with HTTPS to
publish it.

**Exporting.** Read the database directly:

.. code-block:: console

    $ neurolang-feedback-server export --db feedback.sqlite > answers.jsonl
    $ neurolang-feedback-server export --db feedback.sqlite --format examples \
          --engine neurosynth -o neurosynth_feedback.yaml

The JSON lines hold each answer with its steps and ratings, plus a
``training`` pair. That pair is the question with its best query: the latest
corrected query that was checked on the engine, or else the answer's own query
when most ratings say it is correct. ``--format examples`` writes these pairs
as a bank of examples (``assistant_examples``), one per question, ready to be
reviewed and added to an engine's ``<engine>/examples.yaml``.

Evaluation
----------

The `neurolang-text2query-bench
<https://github.com/NeuroLang/neurolang-text2query-bench>`_ repository
benchmarks the assistant in the manner of the `TEXT2SPARQL challenge
<https://text2sparql.aksw.org/latest/>`_. It asks questions on the
``neurosynth`` engine in English and French, and scores the rows of the
generated queries against those of gold queries with precision, recall
and F1.
