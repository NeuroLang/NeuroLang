"""
Semantic similarity between documents and free-text queries.

Documents (for instance scientific articles identified by their PubMed
id) are embedded once with a SPECTER2 document adapter and cached on
disk.  A free-text query is embedded with the SPECTER2 ad-hoc query
adapter and compared to every document by cosine similarity.

The similarities are exposed to NeuroLang as a function built by
:func:`make_similarity_function`, ``query_similarity(query, doc)``, which
returns the similarity of a document to a query text and is used in a
rule as ``study(d) & (sim == query_similarity("visual word form", d))``.

Nothing in this module is specific to a dataset.  Texts come from PubMed
(:func:`fetch_pubmed_abstracts`) or from a CSV/TSV table
(:func:`load_text_table`).  ``torch``, ``transformers`` and ``adapters``
are only imported when the first document or query is embedded; install
them with ``pip install "neurolang[embeddings]"``.

Engines declared in ``engines.yaml`` can opt in through a
``text_embeddings`` section.  Embeddings are computed on the first query
that needs them, or ahead of time with::

    python -m neurolang.utils.text_embeddings -e neurosynth -d neurolang_data
"""

import argparse
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np

LOG = logging.getLogger(__name__)

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EMAIL_ENV = "NEUROLANG_ENTREZ_EMAIL"
API_KEY_ENV = "NCBI_API_KEY"
INSTALL_HINT = 'pip install "neurolang[embeddings]"'

# The files of a model or adapter repository the encoder loads: its
# configuration, tokenizer and PyTorch weights.
MODEL_FILES = ("*.json", "*.txt", "*.bin")

# Written in a local copy of a repository once it is complete.
COMPLETE_MARKER = ".complete"


# ---------------------------------------------------------------------------
# Text sources
# ---------------------------------------------------------------------------


def fetch_pubmed_abstracts(
    pmids, cache_path=None, email=None, api_key=None, batch_size=200
):
    """
    Return the title and abstract of PubMed articles.

    Records are downloaded with the NCBI E-utilities ``efetch`` endpoint
    and stored in a JSON cache, so that only the ids missing from the
    cache are requested.  Requests are rate limited to 3 per second, or
    10 per second when an NCBI API key is given.

    Parameters
    ----------
    pmids : iterable
        PubMed ids; each is converted with ``str``.
    cache_path : str or Path, optional
        JSON file holding the records already downloaded.
    email : str, optional
        Contact e-mail sent to NCBI, defaults to the
        ``NEUROLANG_ENTREZ_EMAIL`` environment variable.  NCBI asks for
        it but does not require it.
    api_key : str, optional
        NCBI API key, defaults to the ``NCBI_API_KEY`` environment
        variable.
    batch_size : int, optional
        Number of ids per request.

    Returns
    -------
    dict
        Maps each id, as a string, to a ``(title, abstract)`` pair.  Both
        are empty strings when PubMed has no record for the id, and the
        abstract is empty when the article has none.

    """
    pmids = list(dict.fromkeys(str(p) for p in pmids))
    records = _read_json(cache_path)
    missing = [p for p in pmids if p not in records]
    email = email or os.environ.get(EMAIL_ENV)
    api_key = api_key or os.environ.get(API_KEY_ENV)
    interval = 0.1 if api_key else 0.34
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        LOG.info("PubMed: fetching %d/%d", start + len(batch), len(missing))
        fetched = _efetch(batch, email, api_key)
        records.update({p: fetched.get(p, ("", "")) for p in batch})
        _write_json(cache_path, records)
        time.sleep(interval)
    return {p: tuple(records[p]) for p in pmids}


def _efetch(pmids, email, api_key):
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "tool": "neurolang",
        "id": ",".join(pmids),
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    data = urllib.parse.urlencode(params).encode()
    return parse_pubmed_xml(_post(EFETCH_URL, data))


def _post(url, data, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, data=data, timeout=120) as res:
                return res.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def parse_pubmed_xml(xml):
    """
    Extract titles and abstracts from a PubMed ``efetch`` XML answer.

    Parameters
    ----------
    xml : bytes or str
        ``PubmedArticleSet`` document.

    Returns
    -------
    dict
        Maps each PMID to a ``(title, abstract)`` pair.  Structured
        abstracts have their sections joined with spaces.

    """
    records = {}
    for article in ET.fromstring(xml).iter("PubmedArticle"):
        citation = article.find("MedlineCitation")
        title = citation.find("Article/ArticleTitle")
        sections = citation.findall("Article/Abstract/AbstractText")
        records[citation.findtext("PMID")] = (
            _text(title),
            " ".join(_text(s) for s in sections),
        )
    return records


def _text(element):
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _read_json(path):
    if path is None or not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _write_json(path, records):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(records, f)
    os.replace(tmp, path)


def load_text_table(path, id_column="id", text_columns=("title", "abstract")):
    """
    Read document texts from a CSV or TSV file.

    Parameters
    ----------
    path : str or Path
        Table with one row per document; ``.tsv`` and ``.tsv.gz`` files
        are tab separated, any other file comma separated.
    id_column : str, optional
        Column holding the document ids.
    text_columns : sequence of str, optional
        The first column is used as the title, the others are joined into
        the body text.

    Returns
    -------
    dict
        Maps each id, as a string, to a ``(title, text)`` pair.

    """
    import pandas as pd

    sep = "\t" if ".tsv" in Path(path).name.lower() else ","
    columns = [id_column, *text_columns]
    df = pd.read_csv(path, sep=sep, usecols=columns, dtype=str)
    df = df.fillna("")
    title, *body = text_columns
    return {
        row[id_column]: (row[title], " ".join(row[c] for c in body).strip())
        for _, row in df.iterrows()
    }


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Specter2Encoder:

    """
    Embed documents and queries with SPECTER2.

    The base model and both adapters are loaded on the first call to
    :meth:`encode_documents` or :meth:`encode_query`.  With a *model_dir*
    they are downloaded once into it (see :meth:`download`) and then
    loaded from there without network; otherwise they come from the
    Hugging Face hub and its cache.  A document is encoded as
    ``title [SEP] abstract`` with the document adapter, a query with the
    query adapter; the embedding is the first ([CLS]) token of the last
    hidden layer.

    Parameters
    ----------
    model : str, optional
        Base model name.
    document_adapter : str, optional
        Adapter used for documents.
    query_adapter : str, optional
        Adapter used for short free-text queries.
    batch_size : int, optional
        Number of documents per forward pass.
    device : str, optional
        Torch device; by default CUDA, then MPS, then CPU.
    model_dir : str or Path, optional
        Directory holding local copies of the model and the adapters.

    """

    def __init__(
        self,
        model="allenai/specter2_base",
        document_adapter="allenai/specter2",
        query_adapter="allenai/specter2_adhoc_query",
        batch_size=16,
        device=None,
        model_dir=None,
    ):
        """Store the configuration; the model is loaded on first use."""
        self.model_name = model
        self.document_adapter = document_adapter
        self.query_adapter = query_adapter
        self.batch_size = batch_size
        self.device = device
        self.model_dir = None if model_dir is None else Path(model_dir)
        self._model = None

    @property
    def cache_key(self):
        """Identify the document embeddings this encoder produces."""
        return json.dumps(
            [self.model_name, self.document_adapter, self.query_adapter]
        )

    def download(self):
        """
        Download the model and the adapters into ``model_dir``, once.

        Returns
        -------
        tuple
            Paths of the local copies of the model, the document adapter
            and the query adapter; their names on the hub without a
            ``model_dir``.

        """
        repos = (self.model_name, self.document_adapter, self.query_adapter)
        if self.model_dir is None:
            return repos
        return tuple(local_model(repo, self.model_dir) for repo in repos)

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"Text embeddings need torch, transformers and adapters: "
                f"{INSTALL_HINT}"
            ) from e
        self._torch = torch
        self.device = self.device or _default_device(torch)
        base, document, query = self.download()
        # Local copies are directories: no hub source, no network.
        source = None if self.model_dir is not None else "hf"
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoAdapterModel.from_pretrained(base)
        model.load_adapter(str(document), source=source, load_as="document")
        model.load_adapter(str(query), source=source, load_as="query")
        self._model = model.to(self.device).eval()
        LOG.info("Loaded %s on %s", self.model_name, self.device)

    def encode_documents(self, titles, texts):
        """
        Embed documents given their titles and body texts.

        Returns
        -------
        numpy.ndarray
            One row per document.

        """
        self._load()
        sep = self._tokenizer.sep_token
        docs = [t + sep + (a or "") for t, a in zip(titles, texts)]
        return self._encode(docs, "document")

    def encode_query(self, text):
        """Embed a free-text query as a vector."""
        self._load()
        return self._encode([text], "query")[0]

    def _encode(self, texts, adapter):
        self._model.set_active_adapters(adapter)
        # Batching texts of similar length reduces padding.
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self._model.config.hidden_size))
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                rows = order[start:start + self.batch_size]
                out[rows] = self._embed_batch([texts[i] for i in rows])
                if len(texts) > self.batch_size:
                    LOG.info("Embedded %d/%d", start + len(rows), len(texts))
        return out

    def _embed_batch(self, texts):
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.device)
        hidden = self._model(**inputs).last_hidden_state
        return hidden[:, 0, :].float().cpu().numpy()


def local_model(repo_id, model_dir):
    """
    Return a local copy of the Hugging Face repository *repo_id*.

    The copy is ``model_dir/<owner>--<name>``.  It is downloaded on the
    first call, then marked complete; later calls return it without
    contacting the hub, so the model works offline.  An interrupted
    download is resumed on the next call.
    """
    path = Path(model_dir) / repo_id.replace("/", "--")
    if (path / COMPLETE_MARKER).exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            f"Downloading {repo_id} needs huggingface_hub: {INSTALL_HINT}"
        ) from e
    LOG.info("Downloading %s to %s", repo_id, path)
    snapshot_download(
        repo_id, local_dir=path, allow_patterns=list(MODEL_FILES)
    )
    (path / COMPLETE_MARKER).touch()
    return path


def _default_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class EmbeddingIndex:

    """
    Cached document embeddings with cosine similarity to queries.

    The embedding matrix is computed on first use (or by :meth:`build`)
    and saved as ``.npz``.  The cache is reused only if it was built with
    the same encoder and the same document ids.

    Parameters
    ----------
    ids : iterable
        Document ids, as stored in the NeuroLang relation.
    load_texts : callable
        ``load_texts(keys)`` returns a mapping from ``str(id)`` to a
        ``(title, text)`` pair; documents missing from it, or with an
        empty title and text, get a zero embedding (similarity 0).
    encoder : object
        Has ``encode_documents(titles, texts)``, ``encode_query(text)``
        and a ``cache_key`` string, e.g. :class:`Specter2Encoder`.
    cache_path : str or Path, optional
        ``.npz`` file holding the embeddings.

    """

    def __init__(self, ids, load_texts, encoder, cache_path=None):
        """Store the ids; embeddings are computed on first use."""
        self.ids = list(dict.fromkeys(ids))
        self._load_texts = load_texts
        self.encoder = encoder
        self.cache_path = None if cache_path is None else Path(cache_path)
        self._matrix = None
        # Per-index cache of the similarities to the 16 most recent
        # queries: it lives and dies with the index.
        self.similarities = lru_cache(maxsize=16)(self._similarities)

    @property
    def cache_key(self):
        """Hash of the encoder and of the document ids."""
        digest = hashlib.sha256(self.encoder.cache_key.encode())
        for d in self.ids:
            digest.update(str(d).encode() + b"\0")
        return digest.hexdigest()

    @property
    def matrix(self):
        """L2-normalised embeddings, one row per document id."""
        if self._matrix is None:
            self._matrix = self._read_cache()
        if self._matrix is None:
            self._matrix = self._embed()
            self._write_cache()
        return self._matrix

    def build(self):
        """Compute, or load from the cache, the document embeddings."""
        return self.matrix

    def _embed(self):
        keys = [str(d) for d in self.ids]
        start = time.perf_counter()
        texts = self._load_texts(keys)
        pairs = [texts.get(k, ("", "")) for k in keys]
        has_text = np.array([bool(t or a) for t, a in pairs])
        LOG.info(
            "Loaded texts in %.1fs: %d documents, %d without abstract, "
            "%d without any text",
            time.perf_counter() - start,
            len(keys),
            sum(not a for _, a in pairs),
            int((~has_text).sum()),
        )
        if not has_text.any():
            raise ValueError("None of the documents has a title or text")
        start = time.perf_counter()
        titles, bodies = zip(*(p for p, h in zip(pairs, has_text) if h))
        embedded = _normalise(self.encoder.encode_documents(titles, bodies))
        matrix = np.zeros((len(keys), embedded.shape[1]), dtype=np.float32)
        matrix[has_text] = embedded
        LOG.info("Embedded documents in %.1fs", time.perf_counter() - start)
        return matrix

    def _read_cache(self):
        if self.cache_path is None or not self.cache_path.exists():
            return None
        with np.load(self.cache_path) as data:
            if str(data["key"]) != self.cache_key:
                LOG.info("Embedding cache %s is stale", self.cache_path)
                return None
            return data["matrix"]

    def _write_cache(self):
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, key=np.array(self.cache_key), matrix=self._matrix)
        os.replace(tmp, self.cache_path)

    def scores(self, query):
        """Cosine similarity of each document, in ``ids`` order, to a query."""
        query_vector = _normalise(self.encoder.encode_query(query)[None])[0]
        return self.matrix @ query_vector

    def _similarities(self, query):
        # Read-only mapping from document id to similarity, so that the
        # cached value cannot be modified by a caller.
        scores = self.scores(query).tolist()
        return MappingProxyType(dict(zip(self.ids, scores)))


def _normalise(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


# ---------------------------------------------------------------------------
# NeuroLang integration
# ---------------------------------------------------------------------------


def make_similarity_function(index, id_type=None):
    """
    Build a NeuroLang function ``f(query, doc) -> similarity``.

    The similarities of every document to a query text are computed once
    and cached by :meth:`EmbeddingIndex.similarities` (the 16 most recent
    queries per index), so each call is a dictionary lookup.

    Parameters
    ----------
    index : EmbeddingIndex
        Index holding the documents.
    id_type : type, optional
        Type of the document ids, by default the type of the first id.

    Returns
    -------
    callable
        ``f(query: str, doc: id_type) -> float``, the cosine similarity
        between the query text and the document, or ``nan`` if ``doc``
        is not in the index.  Use it in a rule as
        ``study(d) & (sim == f("query text", d))``; the query can also be
        a variable bound by a calling rule.

    """
    if id_type is None:
        id_type = type(index.ids[0]) if index.ids else str

    def query_similarity(query, doc):
        return index.similarities(query).get(doc, float("nan"))

    query_similarity.__annotations__ = {
        "query": str, "doc": id_type, "return": float
    }
    return query_similarity


def main(argv=None):
    """
    Download the models and compute the text embeddings of an engine.

    Afterwards, queries embed their texts without network.
    """
    from neurolang_apps.utils.engine_registry import build_engine

    parser = argparse.ArgumentParser(
        prog="python -m neurolang.utils.text_embeddings",
        description=main.__doc__,
    )
    parser.add_argument("--engine", "-e", default="neurosynth")
    parser.add_argument("--data-dir", "-d", default="neurolang_data")
    parser.add_argument("--yaml", default=None, help="Engine YAML file.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    nl = build_engine(
        args.engine,
        Path(args.data_dir),
        yaml_path=None if args.yaml is None else Path(args.yaml),
    )
    for name, index in getattr(nl, "text_embedding_indexes", {}).items():
        # The model is needed by every new query text, even when the
        # document embeddings are already cached.
        index.encoder.download()
        start = time.perf_counter()
        matrix = index.build()
        print(
            f"{name}: {matrix.shape[0]} documents, dimension "
            f"{matrix.shape[1]}, {time.perf_counter() - start:.1f}s, "
            f"cache {index.cache_path}"
        )


if __name__ == "__main__":
    main()
