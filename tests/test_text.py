"""Backend doubles check local-loading policy and batch semantics without downloads."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from xpsych import Compass, Item, ItemBank
from xpsych.text import NLIScorer, EmbeddingScorer


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.shape = self.array.shape

    def to(self, device):
        assert device == 'cpu'
        return self

    def softmax(self, axis):
        e = np.exp(self.array - self.array.max(axis, keepdims=True))
        return Tensor(e / e.sum(axis, keepdims=True))

    def __getitem__(self, key):
        return Tensor(self.array[key])

    def __sub__(self, other):
        return Tensor(self.array - other.array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, hypotheses, **kwargs):
        self.calls.append((texts, hypotheses, kwargs))
        return {'input_ids': Tensor([[len(t), len(h)] for t, h in zip(texts, hypotheses)])}


class Model:
    config = SimpleNamespace(id2label={0: 'neutral', 1: 'contradiction', 2: 'entailment'})
    device = 'cpu'

    def eval(self):
        return self

    def to(self, device):
        assert device == 'cpu'
        return self

    def __call__(self, input_ids):
        a = input_ids.array
        return SimpleNamespace(logits=Tensor(np.column_stack([np.zeros(len(a)), a[:, 1], a[:, 0]])))


def test_nli_batch_order_and_explicit_label_mapping(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    bank = ItemBank([Item('a', 'a', 'xx'), Item('b', 'b', 'xxxx')], {'total': {'a': 1, 'b': 1}})
    tok = Tokenizer()
    scorer = NLIScorer(tok, Model(), batch_size=3)
    result = Compass(bank, scorer).activities(['x', 'xxx', 'xxxxx'])
    expected = []
    for n in [1, 3, 5]:
        row = []
        for h in [2, 4]:
            p = np.exp([0, h, n])
            p /= p.sum()
            row.append(p[2] - p[1])
        expected.append(row)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    assert [len(call[0]) for call in tok.calls] == [3, 3]
    assert all(call[2]['max_length'] == 256 and call[2]['truncation'] for call in tok.calls)
    with pytest.raises(ValueError, match='hypothesis'):
        Compass(ItemBank([Item('a', 'a')], {'total': {'a': 1}}), scorer).activities(['x'])


def test_local_loading_defaults_and_explicit_download(monkeypatch):
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(identifier, **kwargs):
            calls.append((identifier, kwargs))
            return Model() if kwargs.get('use_safetensors') else Tokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=Loader, AutoModelForSequenceClassification=Loader))
    NLIScorer.from_pretrained('local-model', revision='fixed')
    assert all(kwargs['local_files_only'] is True and kwargs['revision'] == 'fixed' for _, kwargs in calls)
    assert calls[-1][1]['use_safetensors'] is True
    NLIScorer.from_pretrained('remote-model', local_files_only=False)
    assert calls[-1][1]['local_files_only'] is False
    enc_calls = []
    def encoder(identifier, **kwargs):
        enc_calls.append(kwargs)
        return object()
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=encoder))
    EmbeddingScorer.from_pretrained('local-model', revision='fixed')
    assert enc_calls[0]['local_files_only'] is True
    assert enc_calls[0]['revision'] == 'fixed'


def test_backend_install_errors_are_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', None)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', None)
    with pytest.raises(ImportError, match=r'xpsych\[text\]'):
        NLIScorer.from_pretrained('missing')
    with pytest.raises(ImportError, match=r'xpsych\[text\]'):
        EmbeddingScorer.from_pretrained('missing')
