"""Tests of SPECTER2's query encoder in numpy, which runs in a browser."""

import numpy as np
import pytest

from neurolang_apps.utils import specter2_numpy as sn

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "visual", "word", "form", "##s",
    "read", "##ing", ",", "(", ")", "cafe", "un", "##known", "中",
]


@pytest.fixture
def tokenizer():
    return sn.WordPieceTokenizer(VOCAB)


def pieces(tokenizer, text, **options):
    return [VOCAB[i] for i in tokenizer.ids(text, **options)]


def test_words_are_lowercased_and_split_on_punctuation(tokenizer):
    assert pieces(tokenizer, "Visual Word, (Form)") == [
        "[CLS]", "visual", "word", ",", "(", "form", ")", "[SEP]"
    ]


def test_words_are_split_in_the_longest_pieces(tokenizer):
    assert pieces(tokenizer, "words reading") == [
        "[CLS]", "word", "##s", "read", "##ing", "[SEP]"
    ]


def test_accents_and_control_characters_are_removed(tokenizer):
    assert pieces(tokenizer, "Café\x00 ­word") == [
        "[CLS]", "cafe", "word", "[SEP]"
    ]


def test_a_word_without_pieces_is_unknown(tokenizer):
    assert pieces(tokenizer, "unknowable visual") == [
        "[CLS]", "[UNK]", "visual", "[SEP]"
    ]


def test_chinese_characters_are_words(tokenizer):
    assert pieces(tokenizer, "word中word") == [
        "[CLS]", "word", "中", "word", "[SEP]"
    ]


def test_long_texts_are_truncated(tokenizer):
    assert pieces(tokenizer, "word " * 10, max_length=5) == [
        "[CLS]", "word", "word", "word", "[SEP]"
    ]


def test_the_tokenizer_is_bert_s():
    transformers = pytest.importorskip("transformers")
    texts = [
        "Visual word form area (VWFA), reading!",
        "naïve Bayes; fMRI-based decoding of 3T BOLD signals",
        "word中word unknowable\ttabs\nnew lines",
    ]
    vocab = sorted(set(VOCAB + [
        "!", ";", "-", "3", "##t", "bo", "##ld", "na", "##ive", "ba",
        "##yes", "f", "##m", "##ri", "based", "de", "##co", "##ding",
        "of", "signal", "area", "v", "##w", "##fa", "tab", "new",
        "line", "##es",
    ]) - {"[PAD]", "[UNK]", "[CLS]", "[SEP]"})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + vocab
    path = pytest.importorskip("pathlib").Path
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        vocab_file = path(directory) / "vocab.txt"
        vocab_file.write_text("\n".join(vocab) + "\n")
        bert = transformers.BertTokenizer(str(vocab_file))
        ours = sn.WordPieceTokenizer(vocab)
        for text in texts:
            assert ours.ids(text) == bert(text)["input_ids"], text


def tiny_weights(rng, vocab_size=len(VOCAB), hidden=8, layers=2,
                 intermediate=16, bottleneck=2, positions=16):
    """The weights of a tiny BERT with a query adapter, at random."""
    def dense(name, n_out, n_in):
        weights[name + ".weight"] = rng.normal(0, .3, (n_out, n_in))
        weights[name + ".bias"] = rng.normal(0, .1, n_out)

    def norm(name):
        weights[name + ".weight"] = 1 + rng.normal(0, .1, hidden)
        weights[name + ".bias"] = rng.normal(0, .1, hidden)

    weights = {
        "embeddings.word_embeddings.weight":
            rng.normal(0, 1, (vocab_size, hidden)),
        "embeddings.position_embeddings.weight":
            rng.normal(0, 1, (positions, hidden)),
        "embeddings.token_type_embeddings.weight":
            rng.normal(0, 1, (2, hidden)),
    }
    norm("embeddings.LayerNorm")
    for n in range(layers):
        p = f"encoder.layer.{n}."
        for part in ("query", "key", "value"):
            dense(p + f"attention.self.{part}", hidden, hidden)
        dense(p + "attention.output.dense", hidden, hidden)
        norm(p + "attention.output.LayerNorm")
        dense(p + "intermediate.dense", intermediate, hidden)
        dense(p + "output.dense", hidden, intermediate)
        norm(p + "output.LayerNorm")
        dense(p + "output.adapter.down", bottleneck, hidden)
        dense(p + "output.adapter.up", hidden, bottleneck)
    return weights


def test_a_saved_encoder_is_loaded_as_it_was(tmp_path):
    weights = tiny_weights(np.random.default_rng(0))
    path = tmp_path / "query.npz"
    sn.save(path, weights, VOCAB, heads=2, dtype=np.float32)
    tokenizer, encoder = sn.load(path)
    same = sn.QueryEncoder(weights, heads=2)
    ids = tokenizer.ids("visual word forms")
    assert np.allclose(encoder(ids), same(ids), atol=1e-6)
    assert encoder(ids).shape == (8,)


def test_float16_weights_are_close(tmp_path):
    weights = tiny_weights(np.random.default_rng(1))
    sn.save(tmp_path / "q16.npz", weights, VOCAB, heads=2)
    tokenizer, encoder = sn.load(tmp_path / "q16.npz")
    exact = sn.QueryEncoder(weights, heads=2)
    ids = tokenizer.ids("reading words")
    a, b = encoder(ids), exact(ids)
    assert a @ b / np.linalg.norm(a) / np.linalg.norm(b) > 0.999


def test_the_encoder_is_the_adapters_model():
    """The numpy encoder gives the [CLS] state of BERT with a sequential
    bottleneck adapter (adapters' ``seq_bn``, SPECTER2's), as torch does."""
    torch = pytest.importorskip("torch")
    adapters = pytest.importorskip("adapters")
    from transformers import BertConfig

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(VOCAB), hidden_size=8, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=16,
    )
    model = adapters.AutoAdapterModel.from_config(config)
    model.add_adapter(
        "query", config=adapters.SeqBnConfig(reduction_factor=4)
    )
    model.set_active_adapters("query")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.3)
    model.eval()
    weights = sn.adapter_model_weights(model.state_dict(), "query")
    tokenizer = sn.WordPieceTokenizer(VOCAB)
    encoder = sn.QueryEncoder(weights, heads=2)
    for text in ("visual word forms", "reading, (cafe)"):
        ids = tokenizer.ids(text)
        with torch.no_grad():
            expected = model(input_ids=torch.tensor([ids])) \
                .last_hidden_state[0, 0].numpy()
        assert np.allclose(encoder(ids), expected, atol=1e-5)
