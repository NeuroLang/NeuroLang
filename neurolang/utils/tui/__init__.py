"""
Interactive terminal REPL for NeuroLang queries.

Started with ``neurolang-query --interactive`` (or ``-i``), it reads
Datalog or SQUALL statements and dot-commands at a prompt and shows the
query results as tables. It needs the optional ``tui`` dependencies
(``pip install "neurolang[tui]"``).

When the standard input is not a terminal, e.g.
``printf 'ans(x) :- study(x)' | neurolang-query -i``, the statements read
from it are run as a batch, split as at the prompt, and the exit status is
1 if one of them failed.

The REPL is organised in modules:

- :mod:`.app`: the REPL loop and query execution,
- :mod:`.commands`: the dot-command registry and the core commands,
- :mod:`.editing`: completion, highlighting, toolbar and editing modes,
- :mod:`.interrupt`: interruption of long queries with Ctrl+C,
- :mod:`.results`: paged display of query results,
- :mod:`.session`: the session's scope and the session commands,
- :mod:`.statements`: splitting of the input into statements.
"""

from .app import InteractiveTuiApp, TuiSettings, main

__all__ = ["InteractiveTuiApp", "TuiSettings", "main"]
