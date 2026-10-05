"""User workflows, published instrument keys, and namespace stability."""
import importlib
import numpy as np
import pytest
import xpsych as xp


def test_public_functions_survive_module_imports():
    for name in ['scoring', 'geometry', 'instruments', 'text']:
        importlib.import_module('xpsych.' + name)
    assert callable(xp.rsa) and callable(xp.compass)


def test_ipip_keys_match_published_counts_and_anchor_items():
    bank = xp.instruments.ipip50()
    assert len(bank.items) == len(set(bank.ids)) == 50
    # Positive/negative counts of the five official 10-item scales.
    counts = [(5, 5), (6, 4), (6, 4), (2, 8), (7, 3)]
    for weights, (pos, neg) in zip(bank.keys.values(), counts):
        assert sum(w == 1 for w in weights.values()) == pos
        assert sum(w == -1 for w in weights.values()) == neg
    assert bank.items[0].text == 'Am the life of the party.'
    assert bank.items[-1].text == 'Do not have a good imagination.'
    assert bank.items[0].hypothesis == 'I am the life of the party.'
    # All equal activities give the published key balances, not Likert totals.
    result = xp.compass(np.ones((2, 50)), bank)
    np.testing.assert_array_equal([s[0] for s in result.values()], [0, 2, 2, -6, 4])
    # No cross-construct leakage; one reverse-keyed item flips only its own score.
    x = np.zeros((1, 50))
    x[0, 5] = .7
    out = xp.compass(x, bank)
    assert out['extraversion'][0] == pytest.approx(-.7)
    assert all(v[0] == 0 for k, v in out.items() if k != 'extraversion')


def test_rsa_row_permutation_and_two_source_inference():
    rng = np.random.default_rng(62)
    x = rng.normal(size=(30, 6))
    y = rng.normal(size=(40, 6))
    d = xp.rsa(x)
    np.testing.assert_allclose(d, xp.rsa(x[::-1]), atol=1e-14)
    result = xp.rsa(x, y, n_permutations=99, seed=3)
    assert result == xp.rsa(d, xp.rsa(y), input_type='rdm', n_permutations=99, seed=3)
    assert -1 <= result.correlation <= 1 and 0 < result.pvalue <= 1
    control = xp.rsa(rng.normal(size=(25, 6)))
    assert xp.rsa(x, y, control=control, method='pearson', n_permutations=19, seed=1) == (
        xp.geometry.compare(d, xp.rsa(y), control=control, method='pearson', n_permutations=19, seed=1))


def test_compass_text_entry_point_uses_real_bank():
    class Scorer:
        def score(self, texts, bank):
            assert len(bank.items) == 50
            return np.ones((len(texts), 50))
    scores = xp.compass(['example narrative', 'second narrative'], xp.instruments.ipip50(), scorer=Scorer())
    np.testing.assert_array_equal(scores['emotional_stability'], [-6, -6])


@pytest.mark.parametrize('kwargs', [
    {'input_type': 'auto'}, {'input_type': 'rdm'}, {'control': np.eye(3)},
])
def test_rsa_rejects_ambiguous_inputs(kwargs):
    with pytest.raises(ValueError):
        xp.rsa(np.eye(3), **kwargs)
