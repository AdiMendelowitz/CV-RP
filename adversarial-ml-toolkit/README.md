# Adversarial ML Toolkit

PyTorch implementations of gradient-based adversarial attacks, with a tested evaluation
pipeline for image classifiers on CIFAR-10.

A ResNet-18 that scores 93.4% on clean CIFAR-10 drops to 35.9% under FGSM noise of
2/255 per pixel, a perturbation invisible to the human eye, and to 12.0% at 16/255. The
grid below shows ten test images attacked at four budgets; green predictions survive the
attack, red ones flip.

![FGSM examples across epsilon](experiments/results/fgsm_examples_by_eps.png)

Those ten are the first samples in dataset order that are classified correctly clean and
flip at 8/255, so they are selected for susceptibility and their per-budget rates are not
population rates. What they do show is that the breaking point is a per-sample property:
the first column's cat survives 2/255 and falls at 4/255, while the rest are already gone
at 2/255. At 16/255 the perturbation is plainly visible as a dither texture, which places
that budget outside the imperceptibility premise and is part of why 8/255 is the field's
standard choice.

| eps | clean accuracy | FGSM accuracy | mean L-inf |
|---------|---------------|---------------|------------|
| 2/255 | 0.9340 | 0.3590 | 0.007843 |
| 4/255 | 0.9340 | 0.2390 | 0.015686 |
| 8/255 | 0.9340 | 0.1660 | 0.031373 |
| 16/255 | 0.9340 | 0.1200 | 0.062745 |

Measured on the first 1,000 CIFAR-10 test images, deterministically. At 2/255, 575 of the
934 samples that were correct before the attack have flipped. The mean L-inf column
averages each sample's largest pixel change, so it reaches the budget as soon as one pixel
saturates and does not report how much of the budget was spent overall.

The full analysis, including the FGSM derivation and the linearity hypothesis behind it,
is in [`notes/adversarial_ml_notes.md`](notes/adversarial_ml_notes.md); raw numbers are in
[`experiments/results/clean_vs_adversarial.csv`](experiments/results/clean_vs_adversarial.csv).

## Attack comparison

All three attacks against the same naturally trained ResNet-18, on the same 1,000
images, at 8/255 for the L-infinity attacks, computed on CPU.

| attack | steps | accuracy | success | mean L-inf | mean L2 |
|--------|-------|----------|---------|------------|---------|
| none   | 0     | 0.9340   | 0.0000  | 0.000000   | 0.000000 |
| FGSM   | 1     | 0.1660   | 0.8223  | 0.031373   | 1.723553 |
| PGD    | 20    | 0.0000   | 1.0000  | 0.031373   | 1.322516 |
| PGD    | 50    | 0.0000   | 1.0000  | 0.031373   | 1.378558 |
| C&W L2 | 100   | 0.0000   | 1.0000  | 0.024777   | 0.230860 |

Raw numbers in
[`experiments/results/robustness_table.csv`](experiments/results/robustness_table.csv).

Three readings matter more than the accuracies themselves.

**Success is not one minus accuracy.** A success here is a sample the model classified
correctly before the attack and incorrectly after it, reported over the 934 samples that
were correct to begin with. Samples the model already got wrong belong to neither count.
The identity that catches a mistake in this definition is that accuracy and success over
a common denominator must sum to clean accuracy: 0.1660 plus 0.7680 is 0.9340.

**PGD wins on direction, not magnitude.** Every L-infinity attack in the table is capped
at the same 8/255, and a perturbation at the full budget on every pixel of a 3x32x32
image has an L2 of 1.7388. FGSM spends 99.1% of that ceiling and still leaves 16.6% of
the images correct; PGD-20 spends 76.1% and leaves none. The advantage of iteration is
where it moves, not how far.

**The C&W row is a different threat model.** It is an L2 attack with no L-infinity
constraint, so its accuracy is not comparable to the rows above it. What it measures is
minimum distortion, which is 5.7 times below PGD-20's L2 and 6.0 times below PGD-50's.
Minimum distortion is also the more informative robustness statistic in general, because
it does not saturate at either end the way accuracy at a fixed budget does. The
implementation fixes the penalty at c = 1 rather than running the paper's per-example
binary search, so 0.230860 is an upper bound on the distortion a full search would find.

## Design

Attacks are pure functions over a model. They compute input gradients with
`torch.autograd.grad` rather than backpropagation into parameters, so they never touch
model state, mutate their inputs, or leave gradient side effects, and they operate
correctly inside `torch.no_grad()` evaluation loops. Perturbation budgets are expressed
once, in `[0, 1]` pixel space; dataset normalisation lives in a `NormalizedModel`
wrapper, so attack code never handles normalisation constants and the budget is measured
in the space the images live in. The test suite enforces this contract for every attack.

Evaluation is deterministic. PGD's random start is seeded per call, so a rerun on the
same device reproduces the CSV byte for byte, and multi-restart PGD seeds each restart
independently so that restart zero is identical to a single-restart run. Reproducibility
is not invariant to batch size, since one call handles one batch.

## Layout

```
attacks/          FGSM, PGD, Carlini-Wagner
defenses/         PGD adversarial training
models/           ResNet-18 (CIFAR stem) and the normalisation wrapper
tests/            pytest suite, including integration tests against the trained model
experiments/      evaluation scripts and their results (CSV, figures)
notes/            derivations and analysis
```

## Reproducing the results

From the toolkit root, with the trained checkpoint and a local CIFAR-10 copy in place:

```powershell
python -m experiments.fgsm_sweep        # the epsilon sweep and its figure
python -m experiments.robustness_eval   # the attack comparison table
```

Paths default to their repository locations and can be overridden with
`CIFAR10_DATA_ROOT` and `CIFAR10_RESNET18_CKPT`. Dataset downloading is intentionally
disabled. Both scripts select CUDA when it is available; the tables above were produced
on CPU, where `robustness_eval` takes a few minutes, dominated by C&W. Accuracies are
device independent, while the norm columns can differ in their last decimals between
devices.

Tests:

```powershell
pytest tests/test_attacks.py -v        # 45 items; slow tests need checkpoint and data
pytest -m "not slow"                   # contract tests only, no checkpoint or data needed
```

The suite collects 45 items after parametrisation over epsilon and targeting: 17 for
FGSM, 16 for PGD, 11 for C&W, and one shared wiring check that confirms the checkpoint
loads and receives inputs in the space it expects. Four are marked slow, and two of those
are empirical rather than contractual, asserting that PGD is at least as strong as FGSM
and that C&W flips a meaningful fraction of a batch. Contract tests constrain shape rather
than strength, so an attack that is subtly weak passes all of them and only an empirical
assertion catches it. The full run takes about 349 seconds on CPU, again dominated by C&W.

## Status

All three attacks are complete, with their behavioural contracts and empirical strength
enforced by the test suite. PGD adversarial training is implemented in
`defenses/adversarial_training.py` and its evaluation is in progress; no robustness
figure for the defended model appears here until that evaluation is complete.

## Limitations

Worth stating plainly, since a robustness number without them is not one to trust.

Every figure above comes from a single evaluation on 1,000 images. At the accuracies in
the table the binomial standard error is under a point, so the comparisons hold, but
differences of a few points elsewhere would not be distinguishable at this sample size.
The 1,000 images are the first 1,000 in dataset order rather than a random sample; on
this model the bias is 0.03 points against the full test set, and it would be larger on
a model whose accuracy sits nearer the middle of the range.

The PGD rows use a single random start. That is immaterial here, since both are already
at zero accuracy and restarts cannot lower them, and it would matter a great deal on a
defended model, where a weak single start overstates robustness.

The C&W row fixes the penalty rather than searching over it, so it reports an upper bound
on minimum distortion rather than the minimum itself.

AutoAttack (Croce and Hein, ICML 2020) is the standard for any robustness claim, and a
hand-rolled PGD number is not a substitute. Nothing in this repository has been
evaluated against it.

## References

Goodfellow, Shlens and Szegedy (2015), Explaining and harnessing adversarial examples,
ICLR 2015. Szegedy et al. (2014), Intriguing properties of neural networks, ICLR 2014.
Madry et al. (2018), Towards deep learning models resistant to adversarial attacks,
ICLR 2018. Carlini and Wagner (2017), Towards evaluating the robustness of neural
networks, IEEE S&P 2017. Athalye, Carlini and Wagner (2018), Obfuscated gradients give
a false sense of security, ICML 2018. Rice, Wong and Kolter (2020), Overfitting in
adversarially robust deep learning, ICML 2020. Croce and Hein (2020), Reliable
evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks,
ICML 2020.