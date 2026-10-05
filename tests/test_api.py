import numpy as np
import pytest
from xpsych import Compass, Item, ItemBank, cosine_activities, stance_activities, aggregate_by_person, geometry as rsa
from xpsych.validation import paired_correlation
from xpsych.text import EmbeddingScorer, nli_indices
from xpsych import _geometry as reference


def bank():
    return ItemBank([Item("a", "An invented positive statement"), Item("b", "An invented negative statement")],
                    {"construct": {"b": -1, "a": 1}})


def test_signed_scores_respect_item_ids_not_key_order():
    result = Compass(bank()).score([[0.8, 0.2], [0.1, 0.9]])
    np.testing.assert_allclose(result["construct"], [0.6, -0.8])


def test_cosine_invariant_to_positive_scaling():
    a = np.array([[1, 0], [1, 1]])
    e = np.array([[1, 1], [-1, 0]])
    np.testing.assert_allclose(cosine_activities(e, a), cosine_activities(e * 8, a * 3))
    np.testing.assert_allclose(cosine_activities(e, a)[0], [2**-0.5, 1])


def test_stance_uses_explicit_class_order():
    p = np.array([[[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]])
    np.testing.assert_allclose(stance_activities(p, entailment_index=1, contradiction_index=0), [[0.6, -0.7]])
    assert nli_indices({0: "CONTRADICTION", 1: "neutral", 2: "entailment"}) == (2, 0)
    with pytest.raises(ValueError):
        nli_indices({0: "LABEL_0", 1: "LABEL_1"})


def test_group_aggregation_preserves_first_seen_ids():
    ids, means = aggregate_by_person([[2, 4], [8, 6], [4, 8]], ["b", "a", "b"])
    assert ids == ("b", "a")
    np.testing.assert_allclose(means, [[3, 6], [8, 6]])


def test_paired_alignment_is_by_id():
    x = {"a": 1, "b": 2, "c": 4, "d": 8}
    y = {"d": 8, "c": 4, "b": 2, "a": 1}
    out = paired_correlation(x, y, n_permutations=99, seed=0)
    assert out.correlation == pytest.approx(1)
    assert out == paired_correlation(x, y, n_permutations=99, seed=0)
    with pytest.raises(ValueError):
        paired_correlation(x, {"a": 1, "b": 2, "c": 3, "e": 4})


def test_geometry_does_not_preserve_person_pairing():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(100, 5))
    order = rng.permutation(len(x))
    np.testing.assert_allclose(rsa.rdm(x), rsa.rdm(x[order]), atol=1e-14)
    correct = paired_correlation(dict(enumerate(x[:, 0])), dict(enumerate(x[:, 0])), seed=0)
    shuffled = paired_correlation(dict(enumerate(x[:, 0])), dict(enumerate(x[order, 0])), seed=0)
    assert correct.correlation > .999
    assert abs(shuffled.correlation) < .25


def test_whitening_and_registered_inference_match_audited_math():
    rng = np.random.default_rng(42)
    x, y, z = [rng.normal(size=(80, 6)) for _ in range(3)]
    c, r = rsa.covariance(x), rsa.covariance(y)
    np.testing.assert_allclose(rsa.whiten(c, r), reference.whiten(c, r), atol=1e-14)
    a, b, control = [np.corrcoef(t.T) for t in (x, y, z)]
    expected = reference.mantel_partial_fl(a, b, control, n_perm=99, rng=13)
    actual = rsa.compare(a, b, control=control, method="pearson", n_permutations=99, seed=13)
    np.testing.assert_allclose([actual.correlation, actual.pvalue], expected, atol=1e-14)
    np.testing.assert_allclose(rsa.whiten(c, c), 0, atol=1e-12)


def test_rank_deficient_reference_is_documented_not_zero():
    r = np.ones((3, 3))
    residual = rsa.whiten(r, r)
    np.testing.assert_allclose(np.linalg.eigvalsh(residual), [-1, -1, 0], atol=1e-7)


def test_spearman_rsa_and_reproducible_randomization():
    rng = np.random.default_rng(19)
    x = rsa.rdm(rng.normal(size=(80, 6)))
    result = rsa.compare(x, x**3, n_permutations=99, seed=10)
    assert result.correlation == pytest.approx(1)
    assert result.pvalue <= .02
    assert result == rsa.compare(x, x**3, n_permutations=99, seed=10)


def test_centering_excludes_current_person():
    out = rsa.center_by_group([[2, 3], [10, 9]], ["q", "q"], ["a", "b"])
    np.testing.assert_allclose(out, [[-8, -6], [8, 6]])
    # Explicit paper fallback for singleton groups.
    np.testing.assert_allclose(rsa.center_by_group([[2, 3]], ["q"], ["a"]), [[0, 0]])


def test_model_independent_text_protocol():
    class Encoder:
        def encode(self, texts):
            return np.array([[len(t), 1] for t in texts], float)
    model = Compass(bank(), EmbeddingScorer(Encoder()))
    activities = model.activities(["invented text", "another example"])
    assert activities.shape == (2, 2)
    np.testing.assert_allclose(model.score_texts(["invented text", "another example"])["construct"],
                               activities[:, 0] - activities[:, 1])


@pytest.mark.parametrize("call", [
    lambda: rsa.rdm([[1, 2], [1, 3]]),
    lambda: rsa.rdm([[1, 2]]),
    lambda: rsa.covariance([[1, np.nan], [2, 3]]),
    lambda: rsa.whiten(np.eye(2), np.zeros((2, 2))),
    lambda: rsa.whiten(np.eye(2), np.diag([1, -1])),
    lambda: rsa.whiten(np.eye(2), np.eye(3)),
    lambda: rsa.whiten(np.eye(2), np.eye(2), floor=0),
    lambda: rsa.compare(np.eye(3), np.eye(3)),
    lambda: cosine_activities([[0, 0]], [[1, 2]]),
    lambda: cosine_activities([[1, 2]], [[1, 2, 3]]),
    lambda: bank().score([[1]]),
    lambda: ItemBank([Item("a", "x"), Item("a", "y")], {"t": {"a": 1}}),
    lambda: ItemBank([Item("a", "x")], {"t": {"missing": 1}}),
    lambda: ItemBank([Item("a", "x")], {"t": {"a": float("nan")}}),
    lambda: ItemBank([Item("a", "x")], {"t": {"a": 0}}),
    lambda: Compass(bank()).activities(["hello"]),
    lambda: aggregate_by_person([[1, 2]], ["a", "b"]),
    lambda: stance_activities([[[.5, .9]]], entailment_index=0, contradiction_index=1),
    lambda: stance_activities([[[.5, .5]]], entailment_index=0, contradiction_index=0),
    lambda: paired_correlation({"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 2, "c": 3}),
])
def test_invalid_inputs_fail_explicitly(call):
    with pytest.raises((ValueError, TypeError)):
        call()


def test_fast_centering_handles_unbalanced_and_singleton_groups():
    rng = np.random.default_rng(45)
    for n in [10, 100, 1000]:
        x = rng.normal(size=(n, 7))
        groups = rng.integers(0, 8, size=n).astype(str)
        people = rng.integers(0, 20, size=n).astype(str)
        np.testing.assert_allclose(rsa.center_by_group(x, groups, people),
                                   reference.center_by_group(x, groups, people), atol=1e-12)


def test_fast_rsa_matches_paper_over_multiple_seeds():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        a, b, c = [np.corrcoef(rng.normal(size=(40, 8)).T) for _ in range(3)]
        for control in [None, c]:
            expected = (reference.mantel(a, b, n_perm=129, rng=seed) if control is None
                        else reference.mantel_partial_fl(a, b, c, n_perm=129, rng=seed))
            actual = rsa.compare(a, b, control=control, method='pearson', n_permutations=129, seed=seed)
            np.testing.assert_allclose([actual.correlation, actual.pvalue], expected, atol=1e-12)


def test_perfect_control_fails_instead_of_spurious_significance():
    rng = np.random.default_rng(9)
    a = np.corrcoef(rng.normal(size=(40, 6)).T)
    with pytest.raises(ValueError, match='residual'):
        rsa.compare(a, a, control=a, method='pearson', seed=0)


@pytest.mark.parametrize('call', [
    lambda: Item('', 'text'),
    lambda: Item('a', ' '),
    lambda: Item('a', 'text', hypothesis=''),
    lambda: ItemBank([], {'a': {'x': 1}}),
    lambda: ItemBank([Item('a', 'text')], {}),
    lambda: ItemBank([Item('a', 'text')], {'': {'a': 1}}),
    lambda: Compass(None),
    lambda: rsa.wording_null([[0, 0], [1, 2]]),
    lambda: rsa.compare(np.ones((3, 4)), np.eye(3)),
    lambda: rsa.compare(np.eye(3), np.eye(4)),
    lambda: rsa.compare(np.eye(3), np.eye(3), method='kendall'),
    lambda: rsa.compare(np.eye(3), np.eye(3), n_permutations=0),
    lambda: rsa.compare(np.eye(3), np.eye(3), control=np.eye(4)),
    lambda: stance_activities([[.1, .9]], entailment_index=0, contradiction_index=1),
    lambda: stance_activities([[[.1, .9]]], entailment_index=-1, contradiction_index=1),
    lambda: paired_correlation([1, 2, 3], [1, 2, 3]),
    lambda: paired_correlation({'a': 1, 'b': 2, 'c': np.nan}, {'a': 1, 'b': 2, 'c': 3}),
])
def test_invalid_metadata_and_inference_inputs(call):
    with pytest.raises((ValueError, TypeError)):
        call()


def test_wording_reference_matches_normalized_gram():
    a = np.array([[2, 0], [1, 1], [0, 3]])
    expected = np.array([[1, 2**-.5, 0], [2**-.5, 1, 2**-.5], [0, 2**-.5, 1]])
    np.testing.assert_allclose(rsa.wording_null(a), expected)


def test_bad_custom_scorer_and_text_inputs():
    class BadScorer:
        def score(self, texts, bank):
            return np.ones((len(texts), 3))
    model = Compass(bank(), BadScorer())
    for texts in ['not a list', [], ['']]:
        with pytest.raises(ValueError):
            model.activities(texts)
    with pytest.raises(ValueError, match='scorer output'):
        model.activities(['example'])
