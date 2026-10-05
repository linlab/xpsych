"""Calibrate fl."""
import json

import numpy as np

import geometry as G
import sim_wording_null as S

REPS_NULL, REPS_POWER, N_PERM = 200, 100, 999

def main():
    rng = np.random.default_rng(20261003)
    A, lab = S.make_items(rng)
    M = A.T @ A
    R = S.make_truth(A, lab, rng)
    out = {"n_perm": N_PERM}
    for n in [189, 86]:
        for g, reps in [(0.0, REPS_NULL), (0.2, REPS_POWER), (0.4, REPS_POWER)]:
            ps = []
            for _ in range(reps):
                m = S.person_means(A, R, n, 155, g, rng)
                W = G.whiten(np.cov(m.T), M)
                ps.append(G.mantel_partial_fl(W, R, M, n_perm=N_PERM, rng=rng)[1])
            out[f"n={n},g={g}"] = dict(reps=reps, share_p_below_005=float(np.mean(np.array(ps) < 0.05)))
            print(n, g, out[f"n={n},g={g}"], flush=True)
    (S.OUT.parent / "fl_calibration.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
