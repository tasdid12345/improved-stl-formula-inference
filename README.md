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

## Reference

This project is motivated by the following paper:

> N Fronda, H Abbas (2022). *Differentiable inference of temporal logic formulas*. … on Computer-Aided Design of Integrated. https://ieeexplore.ieee.org/abstract/document/9852801/

Paper author: Prof. Houssam Abbas, Oregon State University (houssam.abbas@oregonstate.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
