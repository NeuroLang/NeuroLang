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
runs it:

.. code-block:: bash

    neurolang-query --engine neurosynth --squall --ask "Which studies mention pain?" --run

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
    (``neurolang/utils/llm/examples/<engine>.yaml``).

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

The rules of the queries the assistant checks or runs are defined in a
scope that is discarded afterwards. They never stay in the session.

Configuration
-------------

The providers are configured in a YAML file. It is searched for in this
order:

1. the path given by ``--llm-config``;
2. ``$NEUROLANG_LLM_CONFIG``;
3. ``~/.config/neurolang/llm.yaml``;
4. the example shipped in ``neurolang/utils/llm/llm.example.yaml``.

.. code-block:: yaml

    default_provider: albert
    providers:
      albert:                         # Albert, the French State's LLM API
        api: openai
        base_url: https://albert.api.etalab.gouv.fr/v1
        api_key_env: ALBERT_API_KEY
        model: albert-large
        tool_calling: text
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

In the REPL, ``.llm`` shows the configuration. ``.llm provider NAME`` and
``.llm strategy icl|react`` switch the provider and the strategy, and
``.llm reload`` reads the file again. ``.ask --squall``, ``--datalog``,
``--icl`` and ``--react`` choose the language and the strategy for one
question.

.. note::

   The question, the description of the engine's predicates, some of
   their rows and the results of the queries the agent runs are sent to
   the provider.

Evaluation
----------

The `neurolang-text2query-bench
<https://github.com/NeuroLang/neurolang-text2query-bench>`_ repository
benchmarks the assistant in the manner of the `TEXT2SPARQL challenge
<https://text2sparql.aksw.org/latest/>`_. It asks questions on the
``neurosynth`` engine in English and French, and scores the rows of the
generated queries against those of gold queries with precision, recall
and F1.
