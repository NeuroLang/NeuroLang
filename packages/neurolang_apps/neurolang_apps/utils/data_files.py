"""
Files of a data directory fetched when first needed.

Where the data cannot be downloaded from their sources, as in the REPL in
the browser (their servers do not allow calls from web pages), the page's
site has some of the files, and :data:`fetch_missing` fetches them from
there when the engine first needs them (``doc/_static/repl/worker.js``):
the SPECTER2 query model and the embeddings of the studies, and the MNI
template the engine's mask is made from.
"""

from pathlib import Path

# Called with the path of a missing file of the data to fetch it. None:
# nothing is fetched, the data's own fetchers download what they need.
fetch_missing = None


def ensure(path) -> Path:
    """Fetch the file *path* with :data:`fetch_missing` if it is missing,
    and return its path."""
    path = Path(path)
    if fetch_missing is not None and not path.exists():
        fetch_missing(path)
    return path
