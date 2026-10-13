"""
Interactive terminal REPL for NeuroLang queries.

Started with ``neurolang-query --interactive`` (or ``-i``), it reads
Datalog or SQUALL statements and dot-commands at a prompt and shows the
query results as tables. It needs the optional ``tui`` dependencies
(``pip install "neurolang[tui]"``).

The REPL is organised in modules:

- :mod:`.app`: the REPL loop and query execution,
- :mod:`.commands`: the dot-command registry and the core commands,
- :mod:`.results`: conversion and display of query results.
"""

from .app import InteractiveTuiApp, main

__all__ = ["InteractiveTuiApp", "main"]
