"""``python -m serve_repl``: serve the REPL in the browser."""

from . import main

# The builder of the sessions (--users) is a process started anew, which
# imports this module again: it must not serve.
if __name__ == "__main__":
    main()
