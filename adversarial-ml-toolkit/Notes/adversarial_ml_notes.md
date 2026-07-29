# Adversarial Examples: Attacks and Robustness Evaluation

These notes cover the theory behind the toolkit's attacks, the design of their
implementation, and the results of evaluating them against a trained CIFAR-10 classifier.
Every number quoted here is reproduced from the CSV files under
`experiments/results/`, which the sweep and evaluation scripts regenerate
deterministically.

## Background

Szegedy et al. (2014) showed that image classifiers with excellent test accuracy can be
made to misclassify by adding a perturbation small enough to be invisible to a human
viewer. They found these perturbations with box-constrained L-BFGS optimisation and
observed a striking property: adversarial examples crafted against one network often
transfer to networks with different architectures or training data. The initial
explanation appealed to the extreme nonlinearity of deep networks and blind spots left
by finite training data.

## The linearity hypothesis

Goodfellow, Shlens and Szegedy (2015) proposed that locally linear behaviour in
high-dimensional input spaces is sufficient to explain the phenomenon, displacing the
earlier appeal to extreme nonlinearity. Consider a linear score $w^\top \tilde{x}$ with
$\tilde{x} = x + \eta$ and $\lVert \eta \rVert_\infty \le \varepsilon$. Choosing
$\eta = \varepsilon \, \mathrm{sign}(w)$ increases the activation by
$\varepsilon \lVert w \rVert_1$. For an input with $n$ dimensions and average weight
magnitude $m$, that growth is $\varepsilon m n$: it scales with dimensionality even
though no individual pixel moves by more than $\varepsilon$. Many infinitesimal,
coordinated changes add up to a large change in output. Modern networks are built from
components designed to behave linearly over much of their input range, ReLU being the
canonical example, so the same mechanism applies to them. High-dimensional inputs and
locally linear behaviour are sufficient for adversarial vulnerability; no exotic
nonlinearity is required.

## Deriving FGSM

The attacker wants to increase the training loss $J(\theta, x, y)$ subject to an
$L_\infty$ budget $\lVert \eta \rVert_\infty \le \varepsilon$. A first-order expansion
gives

$$J(\theta, x + \eta, y) \approx J(\theta, x, y) + \eta^\top \nabla_x J(\theta, x, y).$$

Maximising the linear term over the $L_\infty$ ball has a closed-form solution: put the
full budget on every coordinate, in the direction of the gradient's sign,

$$\eta^{*} = \varepsilon \, \mathrm{sign}\big(\nabla_x J(\theta, x, y)\big),$$

so the adversarial example is

$$x_{\mathrm{adv}} = \mathrm{clip}_{[0,1]}\big(x + \varepsilon \, \mathrm{sign}(\nabla_x J(\theta, x, y))\big).$$

The targeted variant descends the loss towards a chosen label instead:
$x_{\mathrm{adv}} = \mathrm{clip}_{[0,1]}(x - \varepsilon \, \mathrm{sign}(\nabla_x J(\theta, x, y_{\mathrm{target}})))$.

Two consequences follow directly from the derivation. FGSM costs a single
gradient computation, which makes it the natural baseline attack and the building block
for iterative methods. And the gradient is evaluated at the clean input, so it does not
depend on $\varepsilon$: sweeping the budget moves the adversarial example along one
fixed ray in input space (up to clipping at the pixel-range boundary).

## Implementation design

`attacks/fgsm.py` follows a contract shared across the toolkit's attacks.

The attack is a pure function over a model. It computes the input gradient with
`torch.autograd.grad` on a detached clone rather than `loss.backward()`, so model
parameter gradients are never populated and the model's state is untouched. The body is
wrapped in `torch.enable_grad()` so the attack works inside a `torch.no_grad()`
evaluation loop. The budget is expressed once, in `[0, 1]` pixel space; input
normalisation lives in a `NormalizedModel` wrapper that registers the dataset mean and
standard deviation as buffers and computes $(x - \mu) / \sigma$ in its forward pass.
Normalisation is differentiable, so gradients reach pixel space, rescaled per channel by
$1/\sigma$. Since every $\sigma$ is positive, the rescaling cannot change the sign of any
coordinate, and the sign is the only thing FGSM and PGD consume. C&W consumes the
magnitude, so for that attack the per-channel scaling does affect the search, which is a
further reason to keep normalisation inside the model rather than in the transform
pipeline: the attack then operates on the same pixel-space geometry the budget is
expressed in.

The pytest suite enforces this contract for all three attacks. It collects 45 items after
parametrisation over $\varepsilon$ and targeting: 17 for FGSM, 16 for PGD, 11 for C&W,
and one shared wiring check. The fast tests cover shape and dtype preservation, the
$L_\infty$ budget and pixel range, non-mutation of inputs, absence of parameter-gradient
side effects, correct operation under `no_grad`, restoration of `requires_grad` flags,
and argument validation. Four tests are marked slow and run against the trained network.
Two of those are deliberately split so that a broken model-loading path cannot be
mistaken for a broken attack: one checks clean accuracy in isolation, the other checks
FGSM effectiveness. The remaining two are empirical rather than contractual, asserting
that PGD is at least as strong as FGSM and that C&W flips a meaningful fraction of a
batch. Contract tests constrain shape, not strength, so an attack that is subtly weak
passes all of them; at least one empirical assertion is needed to catch that.

The checkpoint of record is 'best_resnet18_cifar10 (1).pth'. An earlier resnet.py had
regressed to the 7x7 ImageNet stem while this checkpoint was trained with the 3x3 CIFAR
stem; correcting the stem restored a strict state-dict load and reconfirmed 93.43% over
the full test set.

## Empirical evaluation

The target model is a ResNet-18 with the CIFAR stem (3x3 stride-1 first convolution, no
max-pool in the forward pass), which reaches 93.43% on the full CIFAR-10 test set. The
sweep runs FGSM on the first 1,000 test images with no shuffling, so results are exactly
reproducible.

| eps | clean accuracy | FGSM accuracy | mean L-inf |
|---------|---------------|---------------|------------|
| 2/255 | 0.9340 | 0.3590 | 0.007843 |
| 4/255 | 0.9340 | 0.2390 | 0.015686 |
| 8/255 | 0.9340 | 0.1660 | 0.031373 |
| 16/255 | 0.9340 | 0.1200 | 0.062745 |

![FGSM examples across epsilon](../experiments/results/fgsm_examples_by_eps.png)

Several observations follow from the table and the figure.

Most of the damage arrives at the smallest budget. A perturbation of 2/255, genuinely
imperceptible at native resolution and at enlargement, drops accuracy from 93.4% to
35.9%. Of the 934 samples the model classified correctly before the attack, 575 have
already flipped, a rate of 61.6%. The returns then diminish: quadrupling the budget from
4/255 to 16/255 buys another twelve points of degradation, and even at 16/255 some 12% of
samples resist a single gradient step, which is part of the motivation for iterative
attacks such as PGD.

The figure is not evidence about those population rates. Its ten samples are the first
ten in dataset order that are classified correctly clean and flip at 8/255, so they are
selected for susceptibility to this attack. Nine of the ten are already misclassified at
2/255, a rate of 90% against the population's 61.6%, and the gap is the selection rather
than a property of the attack. What the figure does show is per-sample behaviour, which
the table cannot.

Every sample saturates at least one pixel. The mean over samples of the per-sample
maximum perturbation matches $\varepsilon$ to all six reported decimal places at every
budget. That quantity stays at $\varepsilon$ as long as one pixel in each sample takes
the full step without hitting the pixel-range boundary, so it confirms saturation
somewhere in every sample and says nothing about how much of the budget the perturbation
spends overall. The L2 column of the comparison table below is what answers that.

Robustness thresholds are per-sample properties. The first column's cat survives 2/255
and is classified frog at 4/255, so $\varepsilon$ behaves as a budget with
sample-specific breaking points rather than a global switch.

A single ray crosses several class regions. Because the gradient is computed at the
clean input, the four adversarial versions of each sample lie on one straight line
wherever no pixel is clipped at the pixel-range boundary. The fifth column's frog is
classified deer at 2/255 through 8/255 and cat at 16/255, so that line passes through at
least three decision regions. The first two of those transitions happen at budgets that
are invisible; the third does not, since at 16/255 the perturbation is plainly visible as
a dither texture across every panel in the bottom row. That budget therefore sits outside
the imperceptibility premise the threat model rests on, which is part of why 8/255 is the
field's standard choice.

## The saddle-point view and why restarts matter

Madry et al. (2018) frame robust training as a saddle-point problem,

$$\min_\theta \; \mathbb{E}_{(x,y)\sim\mathcal{D}} \Big[ \max_{\delta \in S} L(\theta, x+\delta, y) \Big],$$

where $S$ is the allowed perturbation set, here the $L_\infty$ ball of radius
$\varepsilon$. The inner maximisation is the adversary finding the worst perturbation
of a fixed input; the outer minimisation trains parameters against that worst case.
FGSM and PGD are inner maximisers of this problem, of differing strength: FGSM takes
one gradient step, PGD takes many with projection back into the ball after each. The
PGD evaluation uses a step size of 2/255, following Madry et al.'s CIFAR-10 setup, where
training used 7 steps of size 2 at $\varepsilon = 8$ on the 0 to 255 scale and the
strongest evaluation adversary used 20 steps at the same settings.

The inner problem is not concave, so in principle a single starting point can settle at a
weak local maximum and understate the worst-case loss. Madry et al. investigated this
directly and found the opposite in practice: over $10^5$ random restarts the loss of the
final iterate follows a well-concentrated distribution without extreme outliers, which is
why they report no benefit from restarting PGD within a training batch. Restarts earn
their cost when evaluating a defence rather than when training one. A defence that
obscures gradients rather than removing adversarial examples produces a loss surface
where a single start does get stuck, and reporting robustness from one weak PGD run is
how such defences come to look stronger than they are (Athalye, Carlini and Wagner,
2018). Keeping the strongest result over several random starts guards against that
illusion. For the naturally trained model here a single start already drives accuracy to
zero, so restarts change nothing.

## Minimum distortion and the C&W attack

The attacks above fix a budget and ask how much accuracy falls. Carlini and Wagner (2017)
invert the question: fix the requirement that the prediction change, and ask for the
smallest perturbation that achieves it. Their untargeted L2 attack minimises

$$\lVert \delta \rVert_2^2 + c \cdot f(x + \delta), \qquad
f(x') = \max\big(Z(x')_y - \max_{i \ne y} Z(x')_i, \; -\kappa\big),$$

where $Z$ denotes logits, $y$ the true class, and $\kappa$ a confidence margin left at
zero here. The margin term is negative exactly when the prediction has flipped, so the
optimiser trades distortion against misclassification rather than requiring a hard
constraint. Working on logits rather than on the softmax loss is deliberate: a saturated
softmax gives almost no gradient once the model is confident, which is what defensive
distillation exploited and what this objective defeats.

The box constraint is handled by a change of variables rather than by clipping. The
attack optimises an unconstrained $w$ and evaluates
$x' = \tfrac{1}{2}(\tanh(w) + 1)$, so every iterate lies in $[0, 1]$ by construction and
Adam's momentum is never corrupted by a projection step. Initialising
$w = \mathrm{atanh}(2x - 1)$ makes the first iterate the clean input.

The implementation here departs from the paper in one respect that matters for reading
its results. Carlini and Wagner select $c$ per example with 20 steps of binary search;
`attacks/cw.py` fixes it, and the evaluation below uses $c = 1$ with 100 Adam steps,
keeping for each sample the lowest-distortion iterate that actually flipped. The paper
reports that below roughly $c = 0.1$ the attack rarely succeeds while above roughly
$c = 1$ it always succeeds at the cost of larger perturbations, so $c = 1$ sits at the
useful end of that range and the distortions reported below are an upper bound on what a
binary search would find. Each iterate is scored before the optimiser step that follows
it, so a 100-step run performs 99 scored iterates.

## Attack comparison

Measured on the naturally trained ResNet-18, on CPU, from
`experiments/results/robustness_table.csv`. FGSM and PGD use eps = 8/255 (L-inf);
C&W is an L2 attack with no L-inf budget. Success is a sample classified correctly before
the attack and incorrectly after it, reported over the 934 clean-correct samples; norms are
averaged over those successes. Values are the CSV's, rounded to four decimals.

| attack | steps | accuracy | success | mean L-inf | mean L2 |
|--------|-------|----------|---------|------------|---------|
| none   | 0     | 0.9340   | 0.0000  | 0.0000     | 0.0000  |
| FGSM   | 1     | 0.1660   | 0.8223  | 0.031373   | 1.7236  |
| PGD    | 20    | 0.0000   | 1.0000  | 0.031373   | 1.3225  |
| PGD    | 50    | 0.0000   | 1.0000  | 0.031373   | 1.3786  |
| C&W    | 100   | 0.0000   | 1.0000  | 0.024777   | 0.2309  |

The mean L-inf column is a mean over samples of each sample's maximum pixel change, so it
equals the budget as soon as one pixel saturates. How much of the budget is actually spent
is visible in the L2 column instead. A perturbation sitting at the full budget on every
pixel of a 3x32x32 image has L2 equal to eps * sqrt(3072) = 1.7388. FGSM reaches 99.1% of
that ceiling, saturating almost every coordinate in a single step, exactly as the closed
form solution above predicts. PGD reaches 76.1% at 20 steps and 79.3% at 50, leaving a
quarter of the available perturbation unused while driving accuracy to zero, so iterative
descent is not simply spending more of the budget than FGSM does. C&W reaches the same
complete success at 13.3% of the ceiling, 5.7 times below PGD-20's L2 and 6.0 times below
PGD-50's, which is the reason the attack matters: measuring only accuracy hides that
adversarial examples exist far closer to the clean image than an L-infinity attack reveals.
C&W's mean L-inf sits below 8/255 here only incidentally, since it constrains L2 rather
than L-infinity.

## Reproduction

From the toolkit root:

```powershell
python -m experiments.fgsm_sweep
```

The script expects a local CIFAR-10 copy and the trained checkpoint at their default
repository locations; `CIFAR10_DATA_ROOT` and `CIFAR10_RESNET18_CKPT` override them.
Dataset downloading is intentionally disabled. Outputs are the CSV quoted above and the
figure `experiments/results/fgsm_examples_by_eps.png`.

```powershell
python -m experiments.robustness_eval
```

This writes the attack comparison to `experiments/results/robustness_table.csv`. It takes
a few minutes on CPU, dominated by the C&W optimisation. Both scripts select CUDA when it
is available, and the tables above were produced on CPU. Accuracies are device
independent, while the norm columns can differ in their last decimals between devices, so
the device belongs beside any published table.

## References

Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I. and
Fergus, R. (2014). Intriguing properties of neural networks. ICLR 2014. arXiv:1312.6199.

Goodfellow, I., Shlens, J. and Szegedy, C. (2015). Explaining and harnessing adversarial
examples. ICLR 2015. arXiv:1412.6572.

Carlini, N. and Wagner, D. (2017). Towards evaluating the robustness of neural networks.
IEEE Symposium on Security and Privacy 2017. arXiv:1608.04644.

Madry, A., Makelov, A., Schmidt, L., Tsipras, D. and Vladu, A. (2018). Towards deep
learning models resistant to adversarial attacks. ICLR 2018. arXiv:1706.06083.

Athalye, A., Carlini, N. and Wagner, D. (2018). Obfuscated gradients give a false sense
of security: circumventing defenses to adversarial examples. ICML 2018. arXiv:1802.00420.

Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. Technical
report, University of Toronto.