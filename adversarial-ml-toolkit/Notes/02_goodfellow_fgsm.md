# Explaining and Harnessing Adversarial Examples

Goodfellow, Shlens, Szegedy. ICLR 2015. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)

This paper explains the adversarial examples that Szegedy et al. had observed, and proposes a mechanism opposite to the one first suggested. The argument is that networks are vulnerable because of linear behaviour in high dimensions, not because of non-linear irregularity. For a linear score $w^\top x$, adding a perturbation $\eta$ bounded by $\lVert \eta \rVert_\infty \le \epsilon$ changes the output by up to $w^\top \eta \le \epsilon \lVert w \rVert_1$, with equality when $\eta_i = \epsilon \, \mathrm{sign}(w_i)$. The bound $\epsilon \lVert w \rVert_1$ grows with input dimension while every coordinate moves by no more than $\epsilon$, so imperceptible per-pixel changes accumulate into a large shift in the score.

Applied locally to a network, the gradient $\nabla_x J(\theta, x, y)$ plays the role of $w$, which gives the fast gradient sign method:

$$\eta = \epsilon \cdot \mathrm{sign}\big(\nabla_x J(\theta, x, y)\big).$$

This is a single backward pass, and it is the perturbation that maximises the first-order approximation of the loss inside the $\ell_\infty$ ball of radius $\epsilon$. Its cheapness is the point: a one-step, closed-form attack fools trained networks nearly as effectively as the far more expensive L-BFGS search of the earlier work.

The paper's other contribution is a defence. Training on FGSM-perturbed inputs, or equivalently adding a first-order adversarial term to the loss, improves robustness and gives adversarial training its first practical form. The authors also explain transferability through their mechanism: different models trained on the same task learn similar linear behaviour in the relevant directions, so a perturbation aligned against one tends to align against another. The main caveat, developed by later work, is that a single first-order step underfits the worst case wherever the loss has curvature inside the $\epsilon$ ball, which is why iterative attacks were needed to evaluate robustness honestly.