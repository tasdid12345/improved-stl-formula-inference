# STL Formula Labelling: Window-Feature Gradient Boosting vs. MLP Baseline

Motivation: "Differentiable inference of temporal logic formulas" (2022) proposes a recurrent neural network that learns signal temporal logic (STL) formulas from labelled signals and compares it with other formula-inference methods. This experiment asks a simpler, related question: when the true STL formula is known, how well can standard learners recover the satisfaction labels of signals, and how stable are they across random seeds?

Question: Can a class-balanced, CV-tuned gradient-boosting model that uses STL-style window features match or beat a neural sequence baseline at labelling signals with a known temporal logic formula, and does it do so more stably across seeds?

Dataset: Seeded random-walk signals of length 100, generated with numpy (no download). Labels come from the fixed formula "eventually in [0,20] x>0.6 and always in [0,50] x<0.95", so they are exact. This is not the paper's own dataset (`same_as_paper` is false).

Baseline: an sklearn MLPClassifier on the raw signal, with a small learning-rate and hidden-size grid chosen by 3-fold CV on the training split. It stands in for the paper's recurrent network.

Improved: a class-balanced LightGBM on sliding-window max, min, mean and threshold-crossing-time features, with a small CV-tuned grid. Both models use the same splits, folds, preprocessing and evaluation function. The test set is scored once per seed.

How to run:
```
pip install -r requirements.txt
python analysis.py
```

Results: written to results.json (and figures/accuracy_bar.png).

## Results (latest run)

On simulated random-walk signals with exact STL labels (750 held-out test points per seed, 3 seeds), a standard MLP classifier on raw signals reached 92.8% test accuracy (std 1.3 points), while a class-balanced LightGBM with STL-style window features reached 99.96% (std 0.06 points).

| Metric | Value |
|---|---|
| baseline_score | 0.9276 |
| improved_score | 0.9996 |
| improvement | 0.072 |
| baseline_score_std | 0.0129 |
| improved_score_std | 0.0006285 |
| baseline_f1 | 0.9275 |
| improved_f1 | 0.9996 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated data only, so results do not transfer to the paper's own benchmark. Only 3 seeds and small grids; std is across seeds. The formula is fixed and known, which favours hand-crafted window features. Macro-F1 is a secondary metric. Per-seed positive rate is printed in the log.

## Reference

This project is motivated by the following paper:

> N Fronda, H Abbas (2022). *Differentiable inference of temporal logic formulas*. … on Computer-Aided Design of Integrated. https://ieeexplore.ieee.org/abstract/document/9852801/

Paper author: Prof. Houssam Abbas, Oregon State University (houssam.abbas@oregonstate.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
