# Learned re-ranking of near-optimal block alignments

Motivation: The paper "Near-optimal block alignments" argues that searching for several near-optimal alignments, rather than only the single mathematically optimal one, is useful. Once such a set of candidates exists, something must still choose among them, and the usual default is the candidate with the highest raw alignment score. This experiment checks whether a simple, learned and checkable criterion can choose better.

Question: Can a learned, checkable selection criterion pick better alignments from a set of near-optimal block alignments than the standard choice of highest raw alignment score?

Dataset: Simulated DNA sequence pairs. Random ancestral sequences are mutated (substitutions and short indels) to give two descendants with known true homologous residue pairs. The data are generated with numpy, seed 42. This is not the paper's own dataset.

Baseline: For each pair, a block-based dynamic program (fixed-size blocks, several scoring variants) produces candidate alignments within a threshold of the optimum. The baseline returns the candidate with the highest raw score.

Improved model: The same candidates are re-ranked by a class-balanced logistic regression or LightGBM classifier. It uses per-candidate features (gap count, gap-length spread, block identity, score margin to the optimum). The model is chosen by 5-fold cross-validation on the training pairs only. The test set is used once.

How to run:

    pip install -r requirements.txt
    python analysis.py

Results: results are written to results.json (and figures/accuracy_comparison.png).

## Reference

This project is motivated by the following paper:

> KT Tseng, CB Yang, KS Huang (2008). *Near-optimal block alignments*. IEICE TRANSACTIONS on. https://search.ieice.org/bin/summary.php?id=e91-d_3_789

Paper author: Prof. Chang-Biau Yang, National Sun Yat-sen University (cbyang@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
