"""Precision and batch-order parity for shared paper-scoring computations."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from xpsych import scoring as xs
from xpsych.text import score_nli_pairs
from test_text import Model, Tokenizer

FIXTURES = Path(__file__).parent / 'fixtures'
PAPER = Path(__file__).resolve().parents[1] / 'papers/compass2/codes'


def function_from_source(path, name, **namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_cached_precision_matches_frozen_activity_results():
    data = np.load(FIXTURES / 'activity_reference.npz')
    actual = xs.dot_activities(data['observations'], data['items'])
    assert actual.dtype == np.float32
    # Exact parity with the original expression on this numerical backend.
    np.testing.assert_array_equal(actual, data['observations'] @ data['items'].T)
    # Frozen results were computed on macOS; BLAS implementations can differ
    # by float32 rounding. Keep this separate from exact reduction checks.
    np.testing.assert_allclose(actual, data['activities'], rtol=1e-6, atol=1e-7)
    actual = data['activities']
    for reduction in ['sum', 'mean']:
        np.testing.assert_array_equal(xs.apply_key(actual, data['weights'], reduction=reduction),
                                      data['signed_' + reduction])
    np.testing.assert_array_equal(xs.pool_item_activities(actual, data['owners'], 3), data['pooled'])
    # Re-normalizing float16-rounded caches would change the recorded inputs.
    from xpsych import cosine_activities
    assert not np.array_equal(actual, cosine_activities(data['observations'], data['items']))


@pytest.mark.parametrize('filename,reference,batch,truncation', [
    ('nli_score.py', 'score_symptoms', 64, True),
    ('nli_wai.py', 'score_alliance', 32, 'only_first'),
])
def test_paper_nli_wrappers_match_original_loops(monkeypatch, filename, reference, batch, truncation):
    torch = SimpleNamespace(inference_mode=nullcontext, softmax=lambda tensor, axis: tensor.softmax(axis))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    original = function_from_source(FIXTURES / 'nli_reference.py', reference, np=np, torch=torch)
    current = function_from_source(PAPER / filename, 'score', score_nli_pairs=score_nli_pairs, BATCH=batch)
    # More than one batch; repeated lengths exercise sorting ties and output placement.
    premises = ['a' * n for n in [20, 3, 7, 3, 12, 1, 9, 2, 4, 13, 8, 3, 6, 10, 4, 11, 2]]
    hypotheses = ['b' * n for n in [1, 3, 7, 9, 2]]
    old_tok, new_tok = Tokenizer(), Tokenizer()
    expected = original(old_tok, Model(), 2, 1, premises, hypotheses)
    result = current(new_tok, Model(), 2, 1, premises, hypotheses)
    np.testing.assert_array_equal(result, expected)
    assert old_tok.calls == new_tok.calls
    assert all(call[2]['truncation'] == truncation for call in new_tok.calls)


def test_paper_keyed_means_and_pooling_use_package():
    data = np.load(FIXTURES / 'activity_reference.npz')
    scales = np.array(['Task', 'Bond', 'Goal'] * 3)
    current = function_from_source(PAPER / 'alliance_tests.py', 'keyed', XS=xs,
                                   SCALES=['Task', 'Bond', 'Goal'])
    result = current(data['activities'], data['weights'], scales)
    np.testing.assert_array_equal(result['total'], data['signed_mean'])
    for scale in ['Task', 'Bond', 'Goal']:
        mask = scales == scale
        expected = (data['activities'][:, mask] * data['weights'][mask]).mean(1)
        np.testing.assert_array_equal(result[scale], expected)
    pool = function_from_source(PAPER / 'nli_score.py', 'to_items', pool_item_activities=xs.pool_item_activities)
    np.testing.assert_array_equal(pool(data['activities'], data['owners'], 3), data['pooled'])


@pytest.mark.parametrize('call', [
    lambda: xs.dot_activities([1, 2], [[1, 2]]),
    lambda: xs.dot_activities([[1, np.nan]], [[1, 2]]),
    lambda: xs.dot_activities([[1, 2]], [[1, 2, 3]]),
    lambda: xs.apply_key([[1, 2]], [1]),
    lambda: xs.apply_key([['x']], [1]),
    lambda: xs.apply_key([[np.inf]], [1]),
    lambda: xs.apply_key([[1]], [1], reduction='median'),
    lambda: xs.pool_item_activities([[1, 2]], [0], 2),
    lambda: xs.pool_item_activities([[1, 2]], [0, 0], 2),
    lambda: xs.pool_item_activities([[np.nan]], [0], 1),
])
def test_invalid_scoring_inputs(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize('kwargs', [
    {'truncation': False}, {'entailment_index': -1}, {'entailment_index': 1},
    {'entailment_index': True}, {'premises': 'not a sequence'},
])
def test_invalid_pair_settings(monkeypatch, kwargs):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=nullcontext))
    inputs = dict(premises=['text'], hypotheses=['hypothesis'], entailment_index=2, contradiction_index=1)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        score_nli_pairs(Tokenizer(), Model(), **inputs)
