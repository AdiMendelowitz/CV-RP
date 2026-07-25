# Towards Deep Learning Models Resistant to Adversarial Attacks

Madry, Makelov, Schmidt, Tsipras, Vladu. ICLR 2018. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)

This paper frames adversarial robustness as a single saddle-point problem and derives both an attack and a defence from it. The objective is

$$\min_\theta \; \mathbb{E}_{(x,y)\sim D}\Big[ \max_{\delta \in \mathcal{S}} L(\theta, x+\delta, y) \Big],$$

where $\mathcal{S}$ is the allowed perturbation set, typically the $\ell_\infty$ ball $\lVert \delta \rVert_\infty \le \epsilon$. The inner maximisation is the strongest perturbation an attacker can apply; the outer minimisation trains the weights against that worst case. Fixing $\mathcal{S}$ explicitly turns robustness into a stated specification with a number attached, rather than an informal aspiration.

The inner problem is solved by projected gradient descent (PGD), an iterated version of FGSM. Starting from a random point in the $\epsilon$ ball, each step takes a signed-gradient move and projects back:

$$\delta^{(t+1)} = \Pi_{\mathcal{S}}\Big( \delta^{(t)} + \alpha \, \mathrm{sign}\big(\nabla_\delta L(\theta, x+\delta^{(t)}, y)\big) \Big).$$

The projection is two clamps, one onto $[-\epsilon, \epsilon]^d$ and one keeping $x+\delta$ in valid pixel range. The random start matters: different starts reach comparably high-loss points, which is what licenses treating PGD's output as a reliable worst-case estimate despite the non-convex inner landscape, and distinguishes PGD from FGSM iterated from a fixed point.

The defence is adversarial training with PGD as the inner solver, run for a few steps per batch during training and more steps at evaluation for a faithful measurement. The paper's central empirical claim is that PGD approximates a universal first-order adversary, so a model robust to PGD should resist any gradient-based attack of comparable budget. This held up well: in the subsequent wave of defences that were shown to fail under adaptive attack, PGD adversarial training was the one that survived scrutiny, and it became the field's standard baseline.