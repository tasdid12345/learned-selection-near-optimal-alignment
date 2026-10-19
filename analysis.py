"""Learned re-ranking of near-optimal block alignments vs. highest-raw-score selection.

Simulated DNA pairs with known true alignments -> block-based DP candidates ->
baseline (highest raw score) vs. improved (CV-selected classifier re-ranker).
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

SEED, N_PAIRS, L = 42, 400, 200          # simulation size and ancestor length
SUB, INDEL = 0.15, 0.02                  # substitution prob per lineage, indel prob per site
B, W, THETA = 40, 20, 10.0               # block size, extra Y window, near-optimal threshold
MATCH, MIS, GAP = 1, -1, -2              # reference scoring used for every candidate
SETTINGS = [(1, mm, -g) for mm in (-1, -2) for g in (1, 2, 3, 4)]  # block-DP variants (gap penalties are negative)
SEEDS = (42, 43, 44)


def simulate(rng):
    """Random ancestor mutated into two descendants; returns sequences and true homologous pairs."""
    anc = rng.integers(0, 4, L)
    def mut(b):
        return (b + rng.integers(1, 4)) % 4 if rng.random() < SUB else b
    def ins():
        return list(rng.integers(0, 4, rng.geometric(0.5)))
    X, Y, true = [], [], set()
    for s in range(L):
        r = rng.random()
        if r < INDEL: X += ins()                       # insertion in X
        elif r < 2 * INDEL: Y += ins()                 # insertion in Y
        r = rng.random()
        if r < INDEL: X.append(mut(anc[s]))            # site lost from Y
        elif r < 2 * INDEL: Y.append(mut(anc[s]))      # site lost from X
        else:                                          # homologous column
            X.append(mut(anc[s])); Y.append(mut(anc[s]))
            true.add((len(X) - 1, len(Y) - 1))
    return np.array(X), np.array(Y), true


def block_align(x, y, mt, mm, g):
    """Fixed-size block DP: X is cut into blocks of B; each block is aligned to a Y window
    that starts where the previous block ended. Unaligned residues (leading, internal and
    trailing) are all charged the gap penalty g (g < 0), so the DP objective matches the reference
    score used in features(). Only the end column in Y is chosen freely (best end cell)."""
    pairs, p = [], 0
    for bs in range(0, len(x), B):
        xb = x[bs:bs + B]
        yb = y[p:min(len(y), p + len(xb) + W)]
        k, q = len(xb), len(yb)
        F = [[g * j for j in range(q + 1)]]            # leading Y residues are charged (no free start)
        for i in range(1, k + 1):
            row = [g * i] + [0] * q
            for j in range(1, q + 1):
                s = mt if xb[i - 1] == yb[j - 1] else mm
                row[j] = max(F[i - 1][j - 1] + s, F[i - 1][j] + g, row[j - 1] + g)
            F.append(row)
        jend = max(range(q + 1), key=lambda t: F[k][t])  # end in Y (trailing residues already charged)
        i, j = k, jend
        while i > 0:                                   # traceback
            if j > 0 and F[i][j] == F[i - 1][j - 1] + (mt if xb[i - 1] == yb[j - 1] else mm):
                pairs.append((bs + i - 1, p + j - 1)); i -= 1; j -= 1
            elif F[i][j] == F[i - 1][j] + g: i -= 1    # X residue unaligned
            else: j -= 1                               # Y residue unaligned
        p += jend
    return pairs


def nw_score(x, y):
    """Optimal global alignment score under the reference scoring (linear gaps)."""
    prev = [GAP * j for j in range(len(y) + 1)]
    for i in range(1, len(x) + 1):
        cur = [GAP * i]
        for j in range(1, len(y) + 1):
            s = MATCH if x[i - 1] == y[j - 1] else MIS
            cur.append(max(prev[j - 1] + s, prev[j] + GAP, cur[j - 1] + GAP))
        prev = cur
    return prev[-1]


def features(pairs, x, y, s_opt):
    """Per-candidate features: gap count, gap-length spread, block identity, margin to optimum, size."""
    p = len(pairs)
    matches = int(sum(x[i] == y[j] for i, j in pairs))
    ref = matches * MATCH + (p - matches) * MIS + GAP * (len(x) + len(y) - 2 * p)
    srt = sorted(pairs)
    gaps = [d for (i1, j1), (i2, j2) in zip(srt, srt[1:]) for d in (i2 - i1 - 1, j2 - j1 - 1) if d > 0]
    spread = float(np.std(gaps)) if gaps else 0.0
    return [len(gaps), spread, matches / p, s_opt - ref, p], ref


def build_pair(rng):
    """Generate one simulated pair and its near-optimal candidate set (None if empty)."""
    x, y, true = simulate(rng)
    s_opt = nw_score(x, y)
    cands = {tuple(sorted(block_align(x, y, *s))) for s in SETTINGS}
    rows = []
    for c in sorted(cands):
        if not c: continue
        f, ref = features(c, x, y, s_opt)
        if ref >= s_opt - THETA:                       # near-optimal filter
            rows.append((f, ref, len(set(c) & true)))
    if not rows: return None
    tp = np.array([r[2] for r in rows])
    return (np.array([r[0] for r in rows], float), np.array([r[1] for r in rows]),
            tp / len(true), tp)                        # (features, ref score, accuracy, correct residues)


MODELS = {  # all class-balanced; LightGBM is the only randomized model (seeded)
    "lr_C0.1": lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=s)),
    "lr_C1": lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=s)),
    "lgbm": lambda s: LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=10,
                                     subsample=0.8, subsample_freq=1, class_weight="balanced", random_state=s, verbose=-1),
}


def fit(mk, data, idx, seed):
    """Fit a re-ranker on candidates of the given training pairs; label = candidate with max accuracy."""
    X = np.vstack([data[i][0] for i in idx])
    y = np.concatenate([(data[i][2] == data[i][2].max()).astype(int) for i in idx])
    return mk(seed).fit(X, y)


def pick(m, pair):
    """Index of the candidate with highest predicted probability of being the best one."""
    return int(np.argmax(m.predict_proba(pair[0])[:, 1]))


def main():
    rng = np.random.default_rng(SEED)
    data = [d for d in (build_pair(rng) for _ in range(N_PAIRS)) if d is not None]
    print(f"[sim] kept {len(data)}/{N_PAIRS} pairs with a near-optimal candidate set")
    if len(data) < 50: sys.exit(1)

    # Single split of pairs (no class labels at pair level, so not stratified)
    perm = np.random.default_rng(SEED).permutation(len(data))
    n_test = int(round(0.25 * len(data)))
    test, train = perm[:n_test], perm[n_test:]
    print(f"[split] train={len(train)} test={len(test)}")

    # 5-fold CV on training pairs only to select the re-ranker
    kf = KFold(5, shuffle=True, random_state=SEED)
    cv = {}
    for name, mk in MODELS.items():
        scores = []
        for tr, va in kf.split(train):
            m = fit(mk, data, train[tr], SEED)
            scores.append(np.mean([data[i][2][pick(m, data[i])] for i in train[va]]))
        cv[name] = float(np.mean(scores))
        print(f"[cv] {name}: mean selected accuracy {cv[name]:.4f}")
    best = max(cv, key=cv.get)
    print(f"[cv] selected re-ranker: {best}")

    # Baseline: highest raw score among candidates (deterministic)
    t0 = time.perf_counter()
    base_k = [int(np.argmax(data[i][1])) for i in test]
    base_time = time.perf_counter() - t0
    base_acc = np.array([data[i][2][k] for i, k in zip(test, base_k)])
    base_tp = np.array([data[i][3][k] for i, k in zip(test, base_k)])

    # Improved: selected re-ranker, refit on all training pairs, one run per seed
    runs = []
    for s in SEEDS:
        m = fit(MODELS[best], data, train, s)
        t0 = time.perf_counter()
        ks = [pick(m, data[i]) for i in test]
        dt = time.perf_counter() - t0
        acc = np.array([data[i][2][k] for i, k in zip(test, ks)])
        tp = np.array([data[i][3][k] for i, k in zip(test, ks)])
        runs.append((acc.mean(), tp.mean(), dt))
        print(f"[improved seed {s}] test accuracy {acc.mean():.4f}")

    base_score = float(base_acc.mean())
    imp_scores = [r[0] for r in runs]
    imp_score = float(np.mean(imp_scores))
    print(f"[test] baseline {base_score:.4f} | improved {imp_score:.4f}")

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline\n(top raw score)", f"improved\n({best})"], [base_score, imp_score],
           yerr=[0.0, float(np.std(imp_scores))], color=["grey", "steelblue"], capsize=6)
    ax.set_ylabel("alignment accuracy (test)")
    fig.tight_layout(); fig.savefig("figures/accuracy_comparison.png", dpi=120)

    res = {
        "question": "Can a learned, checkable selection criterion pick better alignments from a set of near-optimal block alignments than the standard choice of highest raw alignment score?",
        "method": "Simulated DNA pairs; block-based DP candidates within a threshold of the optimum; baseline picks highest raw score; improved re-ranks candidates with a CV-selected class-balanced classifier.",
        "dataset": {
            "name": "Simulated ancestor-descendant DNA pairs (numpy, seed 42)",
            "source_url": "numpy random simulation (no download)",
            "n_rows": len(data), "n_train": int(len(train)), "n_test": int(len(test)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "alignment accuracy (fraction of true homologous residue pairs recovered)",
        "higher_is_better": True,
        "baseline": "Highest raw-score candidate among block-DP near-optimal alignments (no explicit selection rule).",
        "improved": f"Same candidates re-ranked by {best} (class-balanced) on gap count, gap-length spread, block identity, score margin and alignment size; chosen by 5-fold CV on training pairs.",
        "metrics": {
            "baseline_score": base_score,
            "improved_score": imp_score,
            "improvement": imp_score - base_score,
            "baseline_score_std": float(np.std([base_score] * len(SEEDS))),
            "improved_score_std": float(np.std(imp_scores)),
            "baseline_correct_residues": float(base_tp.mean()),
            "improved_correct_residues": float(np.mean([r[1] for r in runs])),
            "baseline_select_time_s": base_time,
            "improved_select_time_s": float(np.mean([r[2] for r in runs])),
            "oracle_score": float(np.mean([data[i][2].max() for i in test])),
            "mean_candidates_per_pair": float(np.mean([len(data[i][1]) for i in test])),
            "n_pairs_dropped": float(N_PAIRS - len(data)),
            **{f"cv_{k}": v for k, v in cv.items()},
        },
        "notes": ("Simulated data only; the paper's own dataset and reported result are not used. "
                  "The block DP is a simplified stand-in for the paper's near-optimal block search "
                  "(fixed-size X blocks, semi-global Y windows, candidates from 8 scoring variants). "
                  "Block DP charges leading unaligned Y residues with the gap penalty, matching the "
                  "reference score. The DP gap penalties are now negative (g = -1..-4), as required for "
                  "a penalty; previously they were positive, which rewarded gaps and made every candidate "
                  "fall far below the optimum, so no pair passed the near-optimal filter. "
                  "Labels use the true simulated alignment, which is only available in simulation. "
                  "Near-optimal threshold THETA was fixed a priori, not tuned. "
                  "Baseline is deterministic, so its std is 0 by construction. Split is by pair, not stratified."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("[done] wrote results.json")


if __name__ == "__main__":
    main()
