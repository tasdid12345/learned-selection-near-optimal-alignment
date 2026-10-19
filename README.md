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

## Reference

This project is motivated by the following paper:

> KT Tseng, CB Yang, KS Huang (2008). *Near-optimal block alignments*. IEICE TRANSACTIONS on. https://search.ieice.org/bin/summary.php?id=e91-d_3_789

Paper author: Prof. Chang-Biau Yang, National Sun Yat-sen University (cbyang@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
