# Learned re-ranking of near-optimal block alignments

Question: can a learned, checkable selection criterion pick better alignments from a set of near-optimal block alignments than the highest raw alignment score?

How to run:

    pip install -r requirements.txt
    python analysis.py

The script simulates 400 ancestor-descendant DNA pairs (seed 42), builds block-DP candidate alignments within a near-optimal threshold of the global optimum, and compares two selectors on a held-out 25% of pairs:

- Baseline: highest raw-score candidate.
- Improved: a class-balanced classifier (logistic regression or LightGBM, chosen by 5-fold CV on the training pairs) that re-ranks the same candidates.

Outputs: `results.json` (metrics and notes) and `figures/accuracy_comparison.png`.

Fix note: the block DP now charges leading unaligned Y residues with the gap penalty, consistent with the reference score. Before this, no candidate passed the near-optimal filter, so the script exited with 0 pairs kept.

## Reference

This project is motivated by the following paper:

> KT Tseng, CB Yang, KS Huang (2008). *Near-optimal block alignments*. IEICE TRANSACTIONS on. https://search.ieice.org/bin/summary.php?id=e91-d_3_789

Paper author: Prof. Chang-Biau Yang, National Sun Yat-sen University (cbyang@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
