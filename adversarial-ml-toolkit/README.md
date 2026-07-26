# Adversarial ML Toolkit

PyTorch implementations of gradient-based adversarial attacks, with a tested evaluation
pipeline for image classifiers on CIFAR-10.

A ResNet-18 that scores 93.4% on clean CIFAR-10 drops to 35.9% under FGSM noise of
2/255 per pixel, a perturbation invisible to the human eye, and to 12.0% at 16/255. The
grid below shows the same ten test images attacked at four budgets; green predictions
survive the attack, red ones flip.

![FGSM examples across epsilon](experiments/results/fgsm_examples_by_eps.png)

| eps | clean accuracy | FGSM accuracy | mean L-inf |
|---------|---------------|---------------|------------|
| 2/255 | 0.9340 | 0.3590 | 0.007843 |
| 4/255 | 0.9340 | 0.2390 | 0.015686 |
| 8/255 | 0.9340 | 0.1660 | 0.031373 |
| 16/255 | 0.9340 | 0.1200 | 0.062745 |

Measured on the first 1,000 CIFAR-10 test images, fully deterministically. The full
analysis, including the FGSM derivation and the linearity hypothesis behind it, is in
[`notes/adversarial_ml_notes.md`](notes/adversarial_ml_notes.md); raw numbers are in
[`experiments/results/clean_vs_adversarial.csv`](experiments/results/clean_vs_adversarial.csv).

## Design

Attacks are pure functions over a model. They compute input gradients with
`torch.autograd.grad` rather than backpropagation into parameters, so they never touch
model state, mutate their inputs, or leave gradient side effects, and they operate
correctly inside `torch.no_grad()` evaluation loops. Perturbation budgets are expressed
once, in `[0, 1]` pixel space; dataset normalisation lives in a `NormalizedModel`
wrapper so attack code never handles normalisation constants. The test suite enforces
this contract for each attack as its coverage lands, starting with FGSM.

## Layout

```
attacks/          FGSM, PGD, Carlini-Wagner
models/           ResNet-18 (CIFAR stem) and the normalisation wrapper
tests/            pytest suite, including integration tests against the trained model
experiments/      evaluation scripts and their results (CSV, figures)
notes/            derivations and analysis
```

## Reproducing the results

From the toolkit root, with the trained checkpoint and a local CIFAR-10 copy in place:

```powershell
python -m experiments.fgsm_sweep
```

Paths default to their repository locations and can be overridden with
`CIFAR10_DATA_ROOT` and `CIFAR10_RESNET18_CKPT`. Dataset downloading is intentionally
disabled; evaluation is deterministic, so reruns reproduce the table above exactly.

Tests:

```powershell
pytest tests/test_attacks.py -v        # full suite; slow tests need checkpoint and data
pytest -m "not slow"                   # contract tests only, no checkpoint or data needed
```

## Status

FGSM is complete, with its behavioural contract and effectiveness enforced by the test
suite. PGD and Carlini-Wagner are implemented, with test coverage in progress. An
adversarial training defence and a unified robustness evaluation are the next additions.

## References

Goodfellow, Shlens and Szegedy (2015), Explaining and harnessing adversarial examples,
ICLR 2015. Szegedy et al. (2014), Intriguing properties of neural networks, ICLR 2014.
Madry et al. (2018), Towards deep learning models resistant to adversarial attacks,
ICLR 2018. Carlini and Wagner (2017), Towards evaluating the robustness of neural
networks, IEEE S&P 2017.
