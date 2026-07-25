# Theoretically Principled Trade-off between Robustness and Accuracy

Zhang, Yu, Jiao, Xing, El Ghaoui, Jordan. ICML 2019. [arXiv:1901.08573](https://arxiv.org/abs/1901.08573)

This paper gives the robustness-accuracy trade-off a precise form and derives a training objective from it. The starting point is a decomposition of robust error into two disjoint, exhaustive parts:

$$\mathrm{err}_{\mathrm{rob}} = \mathrm{err}_{\mathrm{nat}} + \mathrm{err}_{\mathrm{bdy}},$$

where natural error is the model misclassifying the clean input, and boundary error is the model classifying the clean input correctly while the decision boundary passes within $\epsilon$ of it. Because a robust mistake must arise in exactly one of these two ways, the relation is an equality rather than a bound. The two terms suggest two jobs: fit the clean data, and push the decision boundary away from the data.

TRADES optimises a surrogate with one term for each. The objective is

$$\min_\theta \; \mathbb{E}\Big[ L(f(x), y) + \tfrac{1}{\lambda} \max_{\lVert \delta \rVert \le \epsilon} \mathrm{KL}\big(f(x) \,\Vert\, f(x+\delta)\big) \Big].$$

The first term is ordinary classification loss for natural error. The second is a KL divergence between the model's prediction on a clean point and on its worst-case neighbour, which targets boundary error by encouraging predictions to stay constant across the $\epsilon$ ball. The regulariser compares the model with itself and uses no label, so it smooths the function even where the label would be wrong, and the perturbation is found by maximising this KL term rather than the classification loss. The weight $1/\lambda$ (written $\beta$ in the released code) is an explicit dial on the trade-off that Madry-style training leaves implicit.

The accomplishment is a defence that is both theoretically grounded and strong in practice. The paper's decomposition is developed into a differentiable upper bound on robust error via classification-calibrated loss, which the surrogate above minimises. The methodology took first place out of roughly 2,000 submissions in the Robust Model Track of the NeurIPS 2018 Adversarial Vision Challenge, and TRADES became a standard adversarial-training method, with $\beta$ giving a tunable, interpretable control over how much clean accuracy is traded for robustness.