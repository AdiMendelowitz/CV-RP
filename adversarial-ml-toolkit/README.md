# Adversarial ML Toolkit

PyTorch implementations of gradient-based adversarial attacks and PGD adversarial training,
with a tested evaluation pipeline for image classifiers on CIFAR-10.

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
is in [`Notes/adversarial_ml_notes.md`](Notes/adversarial_ml_notes.md); raw numbers are in
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

## PGD adversarial training

The same ResNet-18 architecture trained against a 7-step PGD adversary at 8/255 with step
size 2/255, following Madry et al. (2018). Thirty epochs on 45,000 images with 5,000 held
out for checkpoint selection, SGD at lr 0.1 with decays at epochs 15 and 22, seed 0. Two
hours on a T4 at 239 seconds per epoch, peaking at 1.41 GB of GPU memory. The selected
checkpoint is epoch 26 by held-out robust accuracy; the test set was touched once, after
selection.

Reported on the full 10,000-image test set with 95% Wilson intervals.

| model | clean accuracy | robust accuracy (PGD-20, 10 restarts) |
|-------|----------------|---------------------------------------|
| naturally trained | 0.9343 | 0.0000 |
| PGD adversarially trained | 0.7803 [0.7721, 0.7883] | 0.4724 [0.4626, 0.4822] |

The defence buys 47.2 points of robust accuracy for 15.4 points of clean accuracy. Rice,
Wong and Kolter (2020) reach roughly 53% robust and 82% clean with a pre-activation
ResNet-18 at 200 epochs, so this lands 6 and 4 points below at 15% of that budget.

On the 1,000-image subset, for comparison with the natural model row for row:

| attack | steps | accuracy | success | mean L-inf | mean L2 |
|--------|-------|----------|---------|------------|---------|
| none   | 0     | 0.7920   | 0.0000  | 0.000000   | 0.000000 |
| FGSM   | 1     | 0.5250   | 0.3371  | 0.031373   | 1.729898 |
| PGD    | 20    | 0.4660   | 0.4116  | 0.031373   | 1.685636 |
| PGD    | 50    | 0.4620   | 0.4167  | 0.031373   | 1.692547 |
| C&W L2 | 100   | 0.2540   | 0.6793  | 0.165916   | 0.717793 |

### Restarts changed nothing measurable

Ten random restarts, keeping the strongest per sample, moved PGD-20 from 0.4690 to 0.4660
and PGD-50 by the same three images out of 792 attackable. Three samples broke only with
restarts and none broke only without.

This is worth stating because a single random start is the standard way a defence comes to
look stronger than it is, and it does not happen here. A weak start gets stuck when a
defence obscures gradients; adversarial training removes adversarial examples rather than
hiding them, so there is nothing for restarts to expose. The transfer result below says the
same thing from the other direction.

### The evaluation passes all four obfuscated-gradient checks

Athalye, Carlini and Wagner (2018) identify characteristic behaviours of defences that
merely make gradients uninformative. Four are checkable here and all four hold.

An unbounded attack, PGD-50 with the entire pixel box available, drives accuracy to
0.0000. Accuracy is monotone non-increasing in the budget. Iterative beats single-step and
more steps do not hurt: PGD-20 at 0.4660 against FGSM at 0.5250, and PGD-50 at 0.4620.
Black-box transfer does not beat white-box: adversarial examples crafted on the naturally
trained model and evaluated on the defended one leave it at 0.7810, breaking 1.9% of
attackable samples against the white-box attack's 41.2%.

The evaluation additionally recomputes the reported PGD-20 row through an independent code
path and reproduces it to within 1e-12, which is the project's only direct demonstration
that the deterministic cuDNN configuration holds.

### Robustness falls away past the training budget

PGD-20 at ten restarts, across four budgets on the 1,000-image subset:

| eps | defended | naturally trained |
|-----|----------|-------------------|
| 2/255 | 0.7310 | 0.0000 |
| 4/255 | 0.6660 | 0.0000 |
| 8/255 | 0.4660 | 0.0000 |
| 16/255 | 0.1520 | 0.0000 |

The model was trained at 8/255 and keeps 15.2% at double that, so the defence still helps
well outside its training budget while the margin collapses. Robustness from adversarial
training is specific to the epsilon it was trained against, and a figure quoted without its
budget says very little.

### C&W needs three times the distortion

Against the defended model, C&W's mean L2 over 538 broken samples rises to 0.717793 from
the natural model's 0.230860, a factor of 3.11. The median is 0.664148 with an
interquartile range of [0.365128, 1.022333] and a maximum of 2.306616, so the mean is
pulled up by a tail of hard samples and the median is the better single figure. Its mean
L-inf is 0.165916, 5.3 times the 8/255 budget, which is why its accuracy of 0.2540 must
never be read alongside the PGD rows.

Minimum distortion is arguably the better robustness measure precisely here: accuracy at a
fixed budget saturates at both ends, while the distortion an attacker must spend does not.

### PGD works harder against the defence

Against the L2 ceiling of 1.7388, PGD-20 spends 96.9% of the available perturbation on the
defended model where it spent 76.1% on the natural one. The natural model could be broken
from partway into the ball; the defended one forces the attack out to the corners.

### The adversarial predictions do not collapse

Over the 326 subset samples broken by PGD-20 at ten restarts, the predicted classes spread
across all ten labels, the largest being frog at 15.3% against a uniform 10% and the
smallest airplane at 7.4%. The concentration visible in ten FGSM images at 16/255 does not
hold at population scale.

## Design

Attacks are pure functions over a model. They compute input gradients with
`torch.autograd.grad` rather than backpropagation into parameters, so they never touch
model state, mutate their inputs, or leave gradient side effects, and they operate
correctly inside `torch.no_grad()` evaluation loops. Perturbation budgets are expressed
once, in `[0, 1]` pixel space; dataset normalisation lives in a `NormalizedModel`
wrapper, so attack code never handles normalisation constants and the budget is measured
in the space the images live in. The test suite enforces this contract for every attack.

Adversarial examples are generated in eval mode during training, so the attack's forward
passes leave BatchNorm's running statistics untouched and the training-time threat model
matches the evaluation harness. The consequence, expected rather than faulty, is that those
statistics are estimated from the adversarial distribution alone, which is part of why
clean accuracy lands well below the natural model's.

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
experiments/      evaluation scripts, the training notebook, and results (CSV, figures)
Notes/            derivations and analysis
```

## Reproducing the results

From the toolkit root, with the trained checkpoint and a local CIFAR-10 copy in place:

```powershell
python -m experiments.fgsm_sweep        # the epsilon sweep and its figure
python -m experiments.robustness_eval   # the attack comparison table
```

Paths default to their repository locations and can be overridden with
`CIFAR10_DATA_ROOT` and `CIFAR10_RESNET18_CKPT`. Dataset downloading is intentionally
disabled. Both scripts select CUDA when it is available; the natural-model tables were
produced on CPU, where `robustness_eval` takes a few minutes, dominated by C&W. Accuracies
are device independent, while the norm columns can differ in their last decimals between
devices, and the defended tables come from a T4.

Adversarial training runs from `experiments/adversarial_training.ipynb` on a GPU. Two hours
on a T4 for 30 epochs, plus roughly forty minutes for the full evaluation. Every expensive
measurement is cached to disk, so an interrupted session resumes rather than restarts.

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

## Limitations

Worth stating plainly, since a robustness number without them is not one to trust.

Every figure comes from a single training run at one seed. Rice et al. report 0.41 points
of seed variance on their CIFAR-10 robust error, so a difference of a couple of points
between this and another run would mean nothing. A second seed is the single change that
would turn this from a run into a measurement.

Thirty epochs is too short for robust overfitting to appear. Held-out robust accuracy moved
between 0.4639 and 0.4961 from the first decay to the end, a spread inside two standard
errors on 1,024 images, so the phenomenon Rice et al. describe was not observed rather than
absent. Sixty epochs would be needed to look for it.

The natural model's C&W row fixes the penalty rather than searching over it, so it reports
an upper bound on minimum distortion rather than the minimum itself.

AutoAttack (Croce and Hein, ICML 2020) is the standard for any robustness claim, and a
hand-rolled PGD number is not a substitute. Nothing here has been evaluated against it, and
it would be the single most credible addition.

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