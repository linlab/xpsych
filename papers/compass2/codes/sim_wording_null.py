"""Simulate the wording null, interview-script effects and reverse-keyed alliance scores."""
import json
from pathlib import Path

import numpy as np

import geometry as G

import paths as P
OUT = P.RESULTS / "sim_wording_null.json"
D, K_ITEMS, REPS = 128, 25, 200
RIDGES = [0.01, 0.1, 0.3]

def make_items(rng):
    """25 items in 8 wording clusters; three clusters hold one item from each instrument (near-duplicates)."""
    sizes = [5, 4, 3, 3, 3, 3, 2, 2]
    cent = rng.standard_normal((len(sizes), D))
    rows, lab = [], []
    for c, s in enumerate(sizes):
        for _ in range(s):
            rows.append(cent[c] + 0.8 * rng.standard_normal(D)); lab.append(c)
    A = np.array(rows).T
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A, np.array(lab)

def make_truth(A, lab, rng, aligned=0.5):
    """R = corr of a mixture: a wording-aligned part (cluster factors) and a wording-free part (random factors)."""
    k = A.shape[1]
    F_al = np.eye(lab.max() + 1)[lab]
    F_free = rng.standard_normal((k, 3))
    S = aligned * F_al @ F_al.T + (1 - aligned) * F_free @ F_free.T / 3 + 0.3 * np.eye(k)
    return G.cov_to_corr(S)

def person_means(A, R, n_people, n_win, g, rng, sd_person=1.0, sd_win=1.0):
    z = rng.standard_normal((n_people, A.shape[1])) @ np.linalg.cholesky(R).T
    mu = g * z @ A.T + sd_person * rng.standard_normal((n_people, D))
    noise = sd_win * rng.standard_normal((n_people, D)) / np.sqrt(n_win)
    return (mu + noise) @ A

def estimates(m, M):
    C = np.cov(m.T)
    naive = G.cov_to_corr(C)
    out = {"naive": naive, "whitened_N0": G.whiten(C, M)}
    for ridge in RIDGES:
        out[f"deconvolved_ridge{ridge}"] = G.deconvolve_wording(C, M, ridge=ridge)[0]
    return out

def scores(est, R, M):
    return {"r_truth": float(np.corrcoef(G.upper(est), G.upper(R))[0, 1]),
            "r_truth_beyond_wording": G.partial_corr(G.upper(est), G.upper(R), G.upper(M)),
            "r_wording": float(np.corrcoef(G.upper(est), G.upper(M))[0, 1])}

def part1(rng):
    A, lab = make_items(rng)
    M = A.T @ A
    R = make_truth(A, lab, rng)
    out = {"r_truth_vs_wording": float(np.corrcoef(G.upper(R), G.upper(M))[0, 1]),
           "effective_rank_wording": float(np.trace(M) ** 2 / np.sum(M ** 2))}
    grid = {}
    for g in [0.0, 0.1, 0.2, 0.4]:
        for n_people in [86, 189, 275, 1000]:
            for n_win in [155]:
                acc = {}
                for _ in range(REPS):
                    m = person_means(A, R, n_people, n_win, g, rng)
                    for name, est in estimates(m, M).items():
                        for key, v in scores(est, R, M).items():
                            acc.setdefault(name, {}).setdefault(key, []).append(v)
                grid[f"g={g},n={n_people},w={n_win}"] = {
                    name: {key: [float(np.mean(v)), float(np.std(v))] for key, v in d.items()} for name, d in acc.items()}
    out["grid_mean_sd"] = grid

    pv = {"whitened_N0": [], "deconvolved_ridge0.1": []}
    for _ in range(100):
        m = person_means(A, R, 189, 155, 0.0, rng)
        C = np.cov(m.T)
        pv["whitened_N0"].append(G.mantel(G.whiten(C, M), R, control=M, n_perm=199, rng=rng)[1])
        pv["deconvolved_ridge0.1"].append(G.mantel(G.deconvolve_wording(C, M, ridge=0.1)[0], R, control=M, n_perm=199, rng=rng)[1])
    out["null_type1_at_0.05_partial_item_permutation"] = {k: float(np.mean(np.array(v) <= 0.05)) for k, v in pv.items()}
    return out

def part2(rng, n_people=189, h=2.0, reps=50):
    """Adaptive interview script. Question q is about item q (its answers lean towards that item's wording, weight
    h). Ten core questions go to everyone; the other fifteen are follow-ups, asked more often when the person's own
    level on that item is high, as an interviewer probes what came up. Person means then carry the interviewer's
    choices; centring each window on its question's mean from other people leaves how people answer the same
    question. Returns the whitened geometry's association with the truth beyond wording, raw and centred."""
    A, lab = make_items(rng)
    M = A.T @ A
    R = make_truth(A, lab, rng)
    q_off = h * A.T
    res = {}
    for g in [0.0, 0.3]:
        acc = {"raw": [], "centred": []}
        for _ in range(reps):
            z = rng.standard_normal((n_people, K_ITEMS)) @ np.linalg.cholesky(R).T
            mu = g * z @ A.T + rng.standard_normal((n_people, D))
            P, grp, per = [], [], []
            for i in range(n_people):
                follow = 10 + np.flatnonzero(rng.random(15) < 1 / (1 + np.exp(-(1.5 * z[i, 10:] - 0.5))))
                for q in np.concatenate([np.arange(10), follow]):
                    E = mu[i] + q_off[q] + rng.standard_normal((4, D))
                    P.extend(E @ A); grp.extend([q] * 4); per.extend([i] * 4)
            P, grp, per = np.array(P), np.array(grp), np.array(per)
            Pc = G.center_by_group(P, grp, per)
            for name, X in [("raw", P), ("centred", Pc)]:
                m = np.array([X[per == i].mean(0) for i in range(n_people)])
                acc[name].append(G.partial_corr(G.upper(G.whiten(np.cov(m.T), M)), G.upper(R), G.upper(M)))
        res[f"g={g}"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in acc.items()}
    return res

def part3(rng, n_dyads=400, n_win=300, w_neg=0.15, stance_acc=0.8):
    """Reverse keying with the real WAI key: Task 7 forward / 5 reverse, Bond 9 / 3, Goal 6 / 6. Reverse items are
    worded as negations (shared topic, small negation component); a window on a scale's topic carries a stance that
    depends on the dyad's alliance on that scale, and enters the embedding with a small weight. The stance-aware
    scorer reads the stance with accuracy stance_acc. Returns correlations of keyed scale scores with the true
    alliance and with the share of talk about the scale."""
    scales = {"Task": (7, 5), "Bond": (9, 3), "Goal": (6, 6)}
    neg = rng.standard_normal(D); neg /= np.linalg.norm(neg)
    topic = {s: rng.standard_normal(D) for s in scales}
    for s in topic:
        topic[s] /= np.linalg.norm(topic[s])
    items, pol, sc = [], [], []
    for s, (nf, nr) in scales.items():
        for p in [1] * nf + [-1] * nr:
            v = topic[s] + 0.4 * rng.standard_normal(D) / np.sqrt(D) - (p < 0) * 0.5 * neg
            items.append(v / np.linalg.norm(v)); pol.append(p); sc.append(s)
    A = np.array(items).T; pol = np.array(pol); sc = np.array(sc)
    names = list(scales)
    L = np.linalg.cholesky(0.5 * np.ones((3, 3)) + 0.5 * np.eye(3))
    alliance = rng.standard_normal((n_dyads, 3)) @ L.T
    share = rng.dirichlet(np.ones(4) * 2.0, n_dyads)
    res = {s: {"cos": [], "stance": []} for s in names}
    for d in range(n_dyads):
        which = rng.choice(4, n_win, p=share[d])
        E = 1.5 * rng.standard_normal((n_win, D)) / np.sqrt(D)
        stance = np.zeros(n_win)
        for j, s in enumerate(names):
            w = which == j
            stance[w] = np.where(rng.random(w.sum()) < 1 / (1 + np.exp(-1.5 * alliance[d, j])), 1.0, -1.0)
            E[w] += topic[s] - w_neg * (stance[w] < 0)[:, None] * neg
        other = which == 3
        E[other] += rng.standard_normal((other.sum(), D)) / np.sqrt(D)
        read = np.where(rng.random(n_win) < stance_acc, stance, -stance)
        P_cos = E @ A
        P_st = np.clip(P_cos, 0, None) * read[:, None] * pol[None, :]
        for j, s in enumerate(names):
            m = sc == s
            res[s]["cos"].append((P_cos[:, m] * pol[m]).sum(1).mean())
            res[s]["stance"].append((P_st[:, m] * pol[m]).sum(1).mean())
    out = {"cos_forward_vs_reverse_items": float(np.mean([A[:, i] @ A[:, j] for i in range(A.shape[1])
                                                          for j in range(A.shape[1]) if sc[i] == sc[j] and pol[i] != pol[j]]))}
    for j, s in enumerate(names):
        kc, ks = np.array(res[s]["cos"]), np.array(res[s]["stance"])
        out[s] = {"r_keyed_cosine_vs_alliance": float(np.corrcoef(kc, alliance[:, j])[0, 1]),
                  "r_keyed_cosine_vs_share_of_talk": float(np.corrcoef(kc, share[:, j])[0, 1]),
                  "r_keyed_stance_vs_alliance": float(np.corrcoef(ks, alliance[:, j])[0, 1]),
                  "r_keyed_stance_vs_share_of_talk": float(np.corrcoef(ks, share[:, j])[0, 1])}
    return out

def main():
    rng = np.random.default_rng(20261003)
    out = {"part1_population_geometry": part1(rng), "part2_interview_script": part2(rng),
           "part3_reverse_keying": part3(rng)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    p1 = out["part1_population_geometry"]
    print("wording vs truth:", round(p1["r_truth_vs_wording"], 2), "| effective rank of A^T A:", round(p1["effective_rank_wording"], 1))
    for key in ["g=0.0,n=189,w=155", "g=0.1,n=189,w=155", "g=0.2,n=189,w=155", "g=0.4,n=189,w=155", "g=0.2,n=86,w=155", "g=0.2,n=1000,w=155"]:
        row = p1["grid_mean_sd"][key]
        print(key, {n: (round(v["r_truth"][0], 2), round(v["r_truth_beyond_wording"][0], 2), round(v["r_wording"][0], 2)) for n, v in row.items()})
    print("type I (partial, item permutation) at 0.05:", p1["null_type1_at_0.05_partial_item_permutation"])
    print("script:", out["part2_interview_script"])
    print("reverse keying:", out["part3_reverse_keying"])

if __name__ == "__main__":
    main()
