# Towards Evaluating the Robustness of Neural Networks

Carlini, Wagner. IEEE S&P 2017. [arXiv:1608.04644](https://arxiv.org/abs/1608.04644)

This paper introduces a family of strong attacks and uses them to show that defensive distillation, then regarded as a promising defence, provides essentially no robustness. The wider point is methodological: a defence tested only against weak attacks has not been tested, so evaluating robustness requires attacks engineered to be as strong as possible.

The attack minimises perturbation size subject to misclassification, in the same spirit as the earlier L-BFGS formulation but engineered more carefully. Three choices define it. The box constraint is removed by the change of variables $x + \delta = \tfrac{1}{2}(\tanh(w) + 1)$, so an unconstrained optimiser such as Adam can be used on $w$. Success is driven by a margin objective on the pre-softmax logits $Z$,

$$f(x') = \max\Big( \max_{i \ne t} Z(x')_i - Z(x')_t,\; -\kappa \Big),$$

which becomes negative once the target class $t$ leads all others; the confidence parameter $\kappa$ sets how decisively. Working on logits rather than softmax probabilities sidesteps the vanishing gradients that distillation induces, which is why the attack cuts through that defence. The trade-off constant $c$ between distortion and the objective is set by binary search per example, giving a principled minimum-distortion result rather than a hand-tuned one. The attack is instantiated for the $\ell_2$, $\ell_\infty$, and $\ell_0$ norms.

The accomplishment is twofold. Empirically, the $\ell_2$ variant defeats defensive distillation at nearly the perturbation cost of attacking an undefended model, establishing that the defence flattened gradients near the data without moving the decision boundary. Methodologically, the C&W attack became a standard benchmark for years and set the expectation that robustness claims must be made against adaptive, well-optimised attacks. That expectation underlies essentially all serious robustness evaluation since.