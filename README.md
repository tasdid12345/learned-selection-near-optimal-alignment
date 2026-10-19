# Learned re-ranking of near-optimal block alignments

Question: can a learned, checkable selection criterion pick better alignments from a set of near-optimal block alignments than the highest raw alignment score?

## Method
- Simulate 400 ancestor-descendant DNA pairs (numpy, seed 42).
- For each pair, run block-based DP under 8 scoring variants; keep candidates within 10 points of the global optimum.
- Baseline: pick the highest raw-score candidate.
- Improved: re-rank candidates with a class-balanced classifier (logistic regression or LightGBM) chosen by 5-fold CV on the training pairs only.
- Metric: fraction of true homologous residue pairs recovered on a held-out 25% pair split.

## Fix applied
The block DP was given positive gap values (1..4), which rewarded gaps. Every candidate then scored far below the optimum, so no pair passed the near-optimal filter and the script exited. The gap penalties are now negative (-1..-4). Nothing else in the baseline, evaluation, split or test set was changed.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs `results.json` and `figures/accuracy_comparison.png`.

## Results (latest run)

On simulated ancestor-descendant DNA pairs (alignment accuracy, 89 held-out test pairs), the highest-raw-score baseline among near-optimal block alignments reached 82.27%, while the class-balanced logistic re-ranker reached 82.30%, a negligible difference.

| Metric | Value |
|---|---|
| baseline_score | 0.8227 |
| improved_score | 0.823 |
| improvement | 0.0003447 |
| baseline_score_std | 1.11e-16 |
| improved_score_std | 0 |
| baseline_correct_residues | 157.8 |
| improved_correct_residues | 157.8 |
| baseline_select_time_s | 0.0001716 |
| improved_select_time_s | 0.02349 |
| oracle_score | 0.8313 |
| mean_candidates_per_pair | 2.528 |
| n_pairs_dropped | 45 |
| cv_lr_C0.1 | 0.8347 |
| cv_lr_C1 | 0.8347 |
| cv_lgbm | 0.8325 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated data only; the paper's own dataset and reported result are not used. The block DP is a simplified stand-in for the paper's near-optimal block search (fixed-size X blocks, semi-global Y windows, candidates from 8 scoring variants). Block DP charges leading unaligned Y residues with the gap penalty, matching the reference score. The DP gap penalties are now negative (g = -1..-4), as required for a penalty; previously they were positive, which rewarded gaps and made every candidate fall far below the optimum, so no pair passed the near-optimal filter. Labels use the true simulated alignment, which is only available in simulation. Near-optimal threshold THETA was fixed a priori, not tuned. Baseline is deterministic, so its std is 0 by construction. Split is by pair, not stratified.

## Reference

This project is motivated by the following paper:

> KT Tseng, CB Yang, KS Huang (2008). *Near-optimal block alignments*. IEICE TRANSACTIONS on. https://search.ieice.org/bin/summary.php?id=e91-d_3_789

Paper author: Prof. Chang-Biau Yang, National Sun Yat-sen University (cbyang@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
