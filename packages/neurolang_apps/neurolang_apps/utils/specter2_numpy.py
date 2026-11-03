"""
SPECTER2's query encoder in numpy, for where torch is not: a browser.

SPECTER2 embeds a query as the state of the [CLS] token in the last layer
of BERT (SciBERT) with its ad-hoc query adapter, a sequential bottleneck
adapter after the feed-forward block of each layer (``seq_bn`` of the
``adapters`` library). This module computes the same with numpy alone:
:class:`WordPieceTokenizer` splits a text as BERT's tokenizer does, and
:class:`QueryEncoder` runs the model. :func:`save` writes the weights and
the vocabulary in one ``.npz``, read back by :func:`load`; :func:`export`
writes it from the torch model, which it needs. In float16, the default,
the weights take half the space and change no ranking of documents.
"""

import json
import math
import unicodedata

import numpy as np
from scipy.special import erf

# The characters BERT's tokenizer takes as whitespace.
_WHITESPACE = " \t\n\r"


def _is_punctuation(char: str) -> bool:
    """Whether BERT's tokenizer splits words at *char*."""
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or \
            123 <= code <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def _is_chinese(code: int) -> bool:
    """Whether *code* is a CJK ideograph, a word of its own for BERT."""
    return (
        0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF or 0x2F800 <= code <= 0x2FA1F
    )


class WordPieceTokenizer:
    """
    BERT's uncased tokenizer: words, then their longest known pieces.

    A text is cleaned of control characters, its CJK ideographs made
    words of their own, lowercased and stripped of accents, split on
    whitespace and punctuation, and each word split greedily in the
    longest pieces of the vocabulary (``##`` marks a piece continuing a
    word); a word without such pieces is ``[UNK]``.
    """

    def __init__(self, vocab, max_word_length: int = 100):
        self.vocab = {piece: n for n, piece in enumerate(vocab)}
        self.max_word_length = max_word_length

    def _words(self, text: str) -> list:
        """Return the words and punctuation marks of *text*."""
        chars = []
        for char in text:
            code = ord(char)
            if code == 0 or code == 0xFFFD:
                continue
            if char in _WHITESPACE or unicodedata.category(char) == "Zs":
                chars.append(" ")
            elif unicodedata.category(char).startswith("C"):
                continue
            elif _is_chinese(code):
                chars.append(f" {char} ")
            else:
                chars.append(char)
        words = []
        for token in "".join(chars).split():
            token = unicodedata.normalize("NFD", token.lower())
            token = "".join(
                c for c in token if unicodedata.category(c) != "Mn"
            )
            current = ""
            for char in token:
                if _is_punctuation(char):
                    words.extend(w for w in (current, char) if w)
                    current = ""
                else:
                    current += char
            if current:
                words.append(current)
        return words

    def _pieces(self, word: str) -> list:
        """Return the pieces of *word*, or ``[UNK]``."""
        if len(word) > self.max_word_length:
            return ["[UNK]"]
        pieces, start = [], 0
        while start < len(word):
            end = len(word)
            while end > start:
                piece = word[start:end]
                if start > 0:
                    piece = f"##{piece}"
                if piece in self.vocab:
                    break
                end -= 1
            if end == start:
                return ["[UNK]"]
            pieces.append(piece)
            start = end
        return pieces

    def ids(self, text: str, max_length: int = 512) -> list:
        """Return the ids of *text*, between [CLS] and [SEP]."""
        pieces = [p for w in self._words(text) for p in self._pieces(w)]
        body = [self.vocab[p] for p in pieces][:max_length - 2]
        return [self.vocab["[CLS]"], *body, self.vocab["[SEP]"]]


def _layer_norm(x, weight, bias, eps):
    mean = x.mean(-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


class QueryEncoder:
    """
    BERT with a sequential bottleneck adapter, in numpy.

    *weights* are named as in BERT's ``state_dict`` without the ``bert.``
    prefix, those of the adapter of layer ``n`` being
    ``encoder.layer.n.output.adapter.down`` and ``.up``. Calling the
    encoder on token ids returns the [CLS] state of the last layer.
    """

    def __init__(self, weights, heads: int, eps: float = 1e-12):
        self.weights = weights
        self.heads = heads
        self.eps = eps
        self.layers = sum(
            name.endswith("attention.self.query.weight") for name in weights
        )

    def _dense(self, x, name):
        weight = self.weights[name + ".weight"]
        return x @ weight.T.astype(np.float32) + self.weights[name + ".bias"]

    def _norm(self, x, name):
        return _layer_norm(
            x, self.weights[name + ".weight"], self.weights[name + ".bias"],
            self.eps,
        ).astype(np.float32)

    def __call__(self, ids) -> np.ndarray:
        w = self.weights
        n = len(ids)
        x = (
            w["embeddings.word_embeddings.weight"][ids]
            + w["embeddings.position_embeddings.weight"][:n]
            + w["embeddings.token_type_embeddings.weight"][0]
        ).astype(np.float32)
        x = self._norm(x, "embeddings.LayerNorm")
        size = x.shape[1] // self.heads
        for layer in range(self.layers):
            p = f"encoder.layer.{layer}."
            q, k, v = (
                self._dense(x, p + f"attention.self.{part}")
                .reshape(n, self.heads, size).transpose(1, 0, 2)
                for part in ("query", "key", "value")
            )
            scores = q @ k.transpose(0, 2, 1) / math.sqrt(size)
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            scores /= scores.sum(-1, keepdims=True)
            context = (scores @ v).transpose(1, 0, 2).reshape(n, -1)
            x = self._norm(
                self._dense(context, p + "attention.output.dense") + x,
                p + "attention.output.LayerNorm",
            )
            hidden = self._dense(x, p + "intermediate.dense")
            hidden = 0.5 * hidden * (1 + erf(hidden / math.sqrt(2)))
            hidden = self._dense(hidden, p + "output.dense")
            # The adapter reads the normalised output of the block and adds
            # to it the block's output; the block's layer norm follows.
            normed = self._norm(hidden + x, p + "output.LayerNorm")
            down = self._dense(normed, p + "output.adapter.down")
            down = np.maximum(down, 0)
            up = self._dense(down, p + "output.adapter.up")
            x = self._norm(up + hidden + x, p + "output.LayerNorm")
        return x[0]


def save(path, weights, vocab, heads: int, eps: float = 1e-12,
         dtype=np.float16) -> None:
    """Write the *weights*, in *dtype*, and the *vocab* in the ``.npz``
    *path*."""
    config = json.dumps({"heads": heads, "eps": eps})
    np.savez(
        path, vocab=np.array(list(vocab)), config=np.array(config),
        **{name: np.asarray(value, dtype=dtype)
           for name, value in weights.items()},
    )


def load(path):
    """Return the tokenizer and the encoder saved in *path*."""
    with np.load(path) as data:
        config = json.loads(str(data["config"]))
        vocab = data["vocab"].tolist()
        weights = {
            name: data[name] for name in data.files
            if name not in ("vocab", "config")
        }
    return (
        WordPieceTokenizer(vocab),
        QueryEncoder(weights, config["heads"], config["eps"]),
    )


def adapter_model_weights(state_dict, adapter: str) -> dict:
    """
    Return the weights of BERT and of *adapter* in a ``state_dict`` of the
    ``adapters`` library, named as :class:`QueryEncoder` reads them.
    """
    marker = f".output.adapters.{adapter}."
    weights = {}
    for name, value in state_dict.items():
        if not name.startswith("bert.") or ".pooler." in name:
            continue
        name = name[len("bert."):]
        if ".adapters." in name:
            if marker not in name:
                continue
            name = name.replace(
                marker + "adapter_down.0", ".output.adapter.down"
            ).replace(marker + "adapter_up", ".output.adapter.up")
        weights[name] = np.asarray(
            value.detach().cpu().numpy() if hasattr(value, "detach") else value
        )
    return weights


def export(model_path, adapter_path, path, dtype=np.float16) -> None:
    """Write in *path* the model at *model_path* with the query adapter at
    *adapter_path*, as :func:`load` reads it. Needs torch and adapters."""
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    model = AutoAdapterModel.from_pretrained(str(model_path))
    model.load_adapter(str(adapter_path), load_as="query")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    vocab = sorted(tokenizer.vocab, key=tokenizer.vocab.get)
    save(
        path, adapter_model_weights(model.state_dict(), "query"), vocab,
        heads=model.config.num_attention_heads,
        eps=model.config.layer_norm_eps, dtype=dtype,
    )
