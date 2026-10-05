"""Small repeatable CPU benchmark; timings are local measurements, not universal claims."""
import json
import platform
from time import perf_counter
import numpy as np
import xpsych as xp
from xpsych import _geometry as original

rng = np.random.default_rng(42)
windows = rng.normal(size=(10000, 128))
items = rng.normal(size=(25, 128))
profiles = xp.cosine_activities(windows, items)
groups = rng.integers(0, 30, size=len(windows)).astype(str)
people = rng.integers(0, 300, size=len(windows)).astype(str)
a, b, c = [np.corrcoef(rng.normal(size=(300, 25)).T) for _ in range(3)]
bank = xp.ItemBank([xp.Item(str(i), f'Invented item {i}') for i in range(25)],
                {'score': {str(i): 1 for i in range(25)}})

def time_call(fn):
    values = []
    for _ in range(3):
        start = perf_counter()
        result = fn()
        values.append(perf_counter() - start)
    return float(np.median(values)), result

fast_center, x = time_call(lambda: xp.geometry.center_by_group(profiles, groups, people))
old_center, y = time_call(lambda: original.center_by_group(profiles, groups, people))
np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)
fast_rsa, x = time_call(lambda: xp.geometry.compare(a, b, control=c, method='pearson', n_permutations=999, seed=7))
old_rsa, y = time_call(lambda: original.mantel_partial_fl(a, b, c, n_perm=999, rng=7))
np.testing.assert_allclose([x.correlation, x.pvalue], y, atol=1e-12)
score_time, _ = time_call(lambda: xp.Compass(bank).score(profiles))
cosine_time, _ = time_call(lambda: xp.cosine_activities(windows, items))
print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(),
                  'numpy': np.__version__, 'shape': [10000, 128, 25], 'seconds_median_of_three': {
                      'cosine_activities': cosine_time, 'construct_scoring': score_time,
                      'centering_public': fast_center, 'centering_paper': old_center,
                      'rsa_999_public': fast_rsa, 'rsa_999_paper': old_rsa},
                  'agreement': 'rtol=1e-10, atol=1e-12 for centering; atol=1e-12 for RSA'}, indent=2))
