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
    The query is in the prompt: edit it or press Enter to run it.
    nl(neurosynth:SQ)> obtain every Voxel (?i; ?j; ?k) that a Study that mentions 'decision making' reports.

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
    linking). The examples are those closest to the question, chosen by
    BM25 from a bank of checked examples
    (``packages/neurolang_apps/neurolang_apps/utils/llm/examples/<engine>.yaml``).
    An engine inherits the banks of the engines it ``extends``: the
    ``neurosynth`` and ``neurostore`` engines share that of
    ``coordinate_meta_analysis``, and ``neurostore.yaml`` adds questions
    on its concepts, domains and abstracts.

    The query the model writes is then validated on the engine. It is
    parsed, then dry-run; unknown predicates and wrong numbers of
    arguments are reported. When it fails, the formatted error is sent
    back to the model for a correction, up to ``max_repairs`` times.
    With ``samples`` above 1, several queries are generated and the one
    whose result most of them share is kept (self-consistency).

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
    % Links selected studies to the term 'reading'.
    term_association(s, t) :- mentions(s, t), selected_study(s)
    % Calculates the probability that co-activated regions are linked to reading.
    ans(l1, l2, p) :- (l1 < l2), PROB[co_active(l1, l2, s) // term_association(s, 'reading')] = p

Questions given as a description or a Cognitive Atlas concept
-------------------------------------------------------------

On the ``neurosynth`` and ``neurostore`` engines, a topic the vocabulary
has no term for, or a concept of the Cognitive Atlas, is matched to the
studies by the similarity of their title and abstract to its text
(:doc:`tutorial_text_embeddings`). ``similar_study(q, s)`` holds for the
5% of the studies closest to the text ``q``, ``study_title(s, t)`` gives
their titles, and ``cogat_definition(t, d)`` the definition ``d`` of a
Cognitive Atlas concept by its lower-case label ``t``.  Both strategies
know them from the engines' examples and guidance:

.. code-block:: text

    $ neurolang-query --engine neurosynth --ask "Which studies are about the Cognitive Atlas concept response inhibition, with their titles?"
    ans(s, title, v) :- cogat_definition('response inhibition', d), similar_study(d, s), study_similarity(d, s, v), study_title(s, title)

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
      max_steps: 8
      max_repairs: 3
      few_shot_k: 6
      samples: 1
      temperature: 0.0
      timeout: 60
      preview_rows: 10
      verify_rounds: 1       # ReAct: answers back-translation may reject
      comment_rules: true    # a comment line on each rule of the query

The ``api`` of a provider is one of:

``openai``
    The OpenAI chat completions API, spoken by OpenRouter, Albert, vLLM,
    Ollama and most open services.
``anthropic``
    The Anthropic messages API.
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
``neurolang-query --ask``: the question; the query, its explanation, and
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

``.help feedback`` recalls these commands. In the browser, the REPL does not
share its answers.

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
as a bank of examples (``examples/*.yaml``), one per question, ready to be
reviewed and added to an engine's bank.

Evaluation
----------

The `neurolang-text2query-bench
<https://github.com/NeuroLang/neurolang-text2query-bench>`_ repository
benchmarks the assistant in the manner of the `TEXT2SPARQL challenge
<https://text2sparql.aksw.org/latest/>`_. It asks questions on the
``neurosynth`` engine in English and French, and scores the rows of the
generated queries against those of gold queries with precision, recall
and F1.
