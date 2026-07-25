# Intriguing Properties of Neural Networks

Szegedy, Zaremba, Sutskever, Bruna, Erhan, Goodfellow, Fergus. ICLR 2014. [arXiv:1312.6199](https://arxiv.org/abs/1312.6199)

The paper reports two properties of trained deep networks. First, individual units in high-level layers are no more semantically interpretable than random linear combinations of units in the same layer, suggesting that semantic information is encoded in directions through activation space rather than in single units. Second, the finding the paper is remembered for: networks with strong test accuracy misclassify inputs that differ imperceptibly from correctly classified examples.

These adversarial examples are constructed per image by fixing the trained classifier $f$, choosing a target label $l \ne f(x)$, and solving

$$\min_r \; c \lVert r \rVert_2 + \mathrm{loss}_f(x+r,\, l) \quad \text{subject to } x+r \in [0,1]^d$$

with box-constrained L-BFGS. The penalty weight $c$ is found by line search, keeping the smallest value for which the attack succeeds, which approximates the minimum-norm perturbation that crosses the decision boundary. The gradient involved is $\nabla_x \, \mathrm{loss}$ rather than $\nabla_\theta \, \mathrm{loss}$: the weights are frozen, the input is the optimisation variable, and the computation is the same backward pass used in training.

Two experimental results carry the paper. Adversarial examples are found reliably for every model tested, across architectures and activation functions, with perturbations too small to see in the paper's figures. And they transfer: perturbations crafted against one network also fool networks trained with different initialisations, architectures, or disjoint subsets of the training data, so the vulnerability is a property of the learned function class rather than of any one set of weights. Transferability later became the basis of black-box attacks that need no access to the target model's gradients. The authors also observe, in passing, that mixing adversarial examples back into training improves generalisation, an early precursor of adversarial training. They attribute the vulnerability itself to discontinuities in the learned input-output mapping; that explanation was replaced a year later by the linearity hypothesis of Goodfellow, Shlens and Szegedy (2015).