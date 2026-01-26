## Lemma

Below is a fully self-contained rewrite of Lemma 1 + proof that
implements (i) the TV cap, (ii) expected (state-averaged) KL
(TRPO-style), (iii) a critic-error perturbation bound relative to the
true-advantage geometry, and (iv) a general "no-KL" divergence plug-in.
It remains an operator-norm lemma (no Frobenius version).

#### Lemma 1: Decomposition of Learning-Signal Second-Moment Shift (Operator Norm)

#### Setup

Let $\pi_\theta(\cdot\mid s)$ be a stationary stochastic policy, and let
$A_\omega(s,a)$ be an advantage estimator. Define the learning signal
and its second-moment integrand
$$g(s,a;\theta,\omega):=A_\omega(s,a)\nabla_\theta \log \pi_\theta(a\mid s),
\qquad
\phi(s,a;\theta,\omega):=g(s,a;\theta,\omega)g(s,a;\theta,\omega)^\top.$$
Fix an initial-state distribution $\mu$ and $\gamma\in[0,1)$. The
normalized discounted state--action occupancy is $$d^{\pi_\theta}(s,a)
:=(1-\gamma)\sum_{t=0}^\infty \gamma^t\Pr_{\pi_\theta}(s_t=s,a_t=a),$$ a
probability distribution on $\mathcal S\times\mathcal A$. Let
$d^{\pi_\theta}_S$ be its state marginal:
$$d^{\pi_\theta}_S(s):=\int_{\mathcal A} d^{\pi_\theta}(s,a)\,da \quad(\text{or }\sum_a \text{ in discrete actions}).$$
Define the population second-moment ("geometry") matrix
$$G(\theta,\omega)
:=\mathbb{E}_{(s,a)\sim d^{\pi_\theta}}\big[\phi(s,a;\theta,\omega)\big]
=\mathbb{E}_{(s,a)\sim d^{\pi_\theta}}\big[g g^\top\big].$$ For iterates
$(\theta_t,\omega_t)$, write $G_t:=G(\theta_t,\omega_t)$,
$\pi_t:=\pi_{\theta_t}$, $d_t:=d^{\pi_t}$, and $d_{S,t}:=d_S^{\pi_t}$.
We use $\|\cdot\|_2$ for the spectral (operator) norm, and total
variation distance $$D_{TV}(P,Q):=\tfrac12\|P-Q\|_1.$$

#### Assumptions

**A1 (One-step Lipschitz of the integrand, operator norm).** There exist
$L_\theta,L_\omega\ge 0$ such that for all $(s,a)$,
$$\big\|\phi(s,a;\theta_{t+1},\omega_{t+1})-\phi(s,a;\theta_t,\omega_t)\big\|_2
\le
L_\theta\|\theta_{t+1}-\theta_t\|
+
L_\omega\|\omega_{t+1}-\omega_t\|.$$ **A2 (Uniform boundedness along the
step).** There exists $M\ge 0$ such that for all $(s,a)$,
$$\|\phi(s,a;\theta_t,\omega_t)\|_2\le M
\quad\text{and}\quad
\|\phi(s,a;\theta_{t+1},\omega_{t+1})\|_2\le M.$$ (Only boundedness at
the two endpoints is used.)

#### Policy divergence quantities (expected / on-distribution)

Define the state-averaged per-state TV (under the old-policy discounted
state occupancy): $$\overline D_{TV,t}
:=
\mathbb{E}_{s\sim d_{S,t}}\Big[D_{TV}\big(\pi_{t+1}(\cdot\mid s),\pi_t(\cdot\mid s)\big)\Big].$$
Define the state-averaged KL (again under $d_{S,t}$):
$$\overline D_{KL,t}
:=
\mathbb{E}_{s\sim d_{S,t}}\Big[D_{KL}\big(\pi_t(\cdot\mid s)\,||\,\pi_{t+1}(\cdot\mid s)\big)\Big].$$

#### Statement (one-step shift + divergence generality)

Under A1--A2, $$\boxed{
\|G_{t+1}-G_t\|_2
\le
L_\theta\|\theta_{t+1}-\theta_t\|
+
L_\omega\|\omega_{t+1}-\omega_t\|
+
2M\cdot \min\left\{1,\ \frac{\overline D_{TV,t}}{1-\gamma}\right\}.
}
\tag{L1-TV}$$ Moreover, this last term admits a divergence plug-in: if
there exist a per-state divergence $D(\cdot||\cdot)$ and a nondecreasing
function $\Psi:[0,\infty)\to[0,1]$ such that for all $s$,
$$D_{TV}\big(\pi_{t+1}(\cdot\mid s),\pi_t(\cdot\mid s)\big)
\le
\Psi\Big(D\big(\pi_t(\cdot\mid s)\,||\,\pi_{t+1}(\cdot\mid s)\big)\Big),
\tag{TV-compare}$$ then $$\boxed{
\|G_{t+1}-G_t\|_2
\le
L_\theta\|\theta_{t+1}-\theta_t\|
+
L_\omega\|\omega_{t+1}-\omega_t\|
+
2M\cdot \min\left\{1,\ \frac{1}{1-\gamma}
\mathbb{E}_{s\sim d_{S,t}}\big[\Psi(D_s)\big]\right\},
}
\tag{L1-general}$$ where $D_s:=D(\pi_t(\cdot|s)||\pi_{t+1}(\cdot|s))$.

**KL instantiation (Pinsker + Jensen).** Using Pinsker,
$$D_{TV}(\pi_{t+1}(\cdot|s),\pi_t(\cdot|s))
\le \sqrt{\tfrac12 D_{KL}(\pi_t(\cdot|s)||\pi_{t+1}(\cdot|s))},$$ and
Jensen (concavity of $\sqrt{\cdot}$) yields
$$\overline D_{TV,t}\le \sqrt{\tfrac12 \overline D_{KL,t}}.$$ Thus from
(L1-TV), $$\boxed{
\|G_{t+1}-G_t\|_2
\le
L_\theta\|\theta_{t+1}-\theta_t\|
+
L_\omega\|\omega_{t+1}-\omega_t\|
+
\min\left\{2M,\ \frac{M\sqrt2}{1-\gamma}\sqrt{\overline D_{KL,t}}\right\}.
}
\tag{L1-KL}$$

#### Critic-error perturbation (algorithmic vs true-advantage geometry)

Let $A^{\pi_\theta}(s,a)$ denote the true advantage for policy
$\pi_\theta$. Define the "ideal" signal and matrix
$$g_\star(s,a;\theta):=A^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s),
\qquad
G_\star(\theta):=\mathbb{E}_{(s,a)\sim d^{\pi_\theta}}\big[g_\star g_\star^\top\big].$$
Define critic error and induced signal error
$$e_{\theta,\omega}(s,a):=A_\omega(s,a)-A^{\pi_\theta}(s,a),
\qquad
\delta g(s,a;\theta,\omega):=e_{\theta,\omega}(s,a)\nabla_\theta\log\pi_\theta(a|s).$$
Assume square-integrability under $d^{\pi_\theta}$:
$\mathbb{E}\|g_\star\|_2^2<\infty$ and
$\mathbb{E}\|\delta g\|_2^2<\infty$. Define
$$\sigma(\theta)^2:=\mathbb{E}_{d^{\pi_\theta}}\|g_\star\|_2^2,
\qquad
\varepsilon(\theta,\omega)^2:=\mathbb{E}_{d^{\pi_\theta}}\|\delta g\|_2^2
= \mathbb{E}_{d^{\pi_\theta}}\Big[e_{\theta,\omega}(s,a)^2 \|\nabla_\theta\log\pi_\theta(a|s)\|_2^2\Big].$$
Then $$\boxed{
\|G(\theta,\omega)-G_\star(\theta)\|_2
\le
2\sigma(\theta)\varepsilon(\theta,\omega)
+
\varepsilon(\theta,\omega)^2.
}
\tag{critic-pert}$$ This is the clean "critic perturbation" term you can
add as part of the perturbation budget in Davis--Kahan--type arguments.

#### Proof of Lemma 1 (shift bound)

Write $\phi_t(s,a):=\phi(s,a;\theta_t,\omega_t)$ and
$\phi_{t+1}(s,a):=\phi(s,a;\theta_{t+1},\omega_{t+1})$, and recall
$d_t=d^{\pi_t}$, $d_{t+1}=d^{\pi_{t+1}}$.

**Step 1: Add--subtract decomposition (integrand shift vs measure
shift)** $$\begin{aligned}
G_{t+1}-G_t
&=\mathbb{E}_{(s,a)\sim d_{t+1}}[\phi_{t+1}(s,a)]
-\mathbb{E}_{(s,a)\sim d_t}[\phi_t(s,a)] \\
&=
\underbrace{\Big(\mathbb{E}_{d_{t+1}}[\phi_{t+1}]
-\mathbb{E}_{d_{t+1}}[\phi_t]\Big)}_{=:T_{\mathrm{param}}}
+
\underbrace{\Big(\mathbb{E}_{d_{t+1}}[\phi_t]
-\mathbb{E}_{d_t}[\phi_t]\Big)}_{=:T_{\mathrm{occ}}}.
\end{aligned}$$ By the triangle inequality,
$$\|G_{t+1}-G_t\|_2 \le \|T_{\mathrm{param}}\|_2+\|T_{\mathrm{occ}}\|_2.$$

**Step 2: Bound the parametric shift term ($T_{\mathrm{param}}$)** Using
$\|\mathbb{E}[X]\|_2 \le \mathbb{E}[\|X\|_2]$,
$$\|T_{\mathrm{param}}\|_2
= \left\|\mathbb{E}_{d_{t+1}}[\phi_{t+1}-\phi_t]\right\|_2
\le
\mathbb{E}_{d_{t+1}}\big[\|\phi_{t+1}-\phi_t\|_2\big].$$ By A1,
$\|\phi_{t+1}-\phi_t\|_2 \le L_\theta\|\theta_{t+1}-\theta_t\|+L_\omega\|\omega_{t+1}-\omega_t\|$
pointwise, hence $$\|T_{\mathrm{param}}\|_2
\le
L_\theta\|\theta_{t+1}-\theta_t\|
+
L_\omega\|\omega_{t+1}-\omega_t\|.$$

**Step 3: Bound the occupancy (distribution) shift term
($T_{\mathrm{occ}}$)** Write the difference of expectations as an
integral against the signed measure $d_{t+1}-d_t$: $$T_{\mathrm{occ}}
= \int_{\mathcal S\times\mathcal A} \phi_t(s,a)\big(d_{t+1}-d_t\big)(ds,da).$$
Using the standard bound $\|\int X\,d\nu\|_2 \le \int \|X\|_2\,d|\nu|$
and A2, $$\|T_{\mathrm{occ}}\|_2
\le
\int \|\phi_t(s,a)\|_2\,|d_{t+1}-d_t|(ds,da)
\le
M\|d_{t+1}-d_t\|_1
= 2M D_{TV}(d_{t+1},d_t).$$ Thus it remains to bound
$D_{TV}(d_{t+1},d_t)$ in terms of a policy divergence.

**Step 4: Occupancy-TV bound via expected per-state TV (with TV cap)**

#### Lemma (Discounted occupancy TV controlled by expected per-state policy TV)

For any two stationary policies $(\pi,\pi')$ (sharing the same initial
$\mu$), $$D_{TV}(d^{\pi'},d^\pi)
\le
\min\left\{1,\ \frac{1}{1-\gamma}
\mathbb{E}_{s\sim d_S^\pi}\big[D_{TV}(\pi'(\cdot|s),\pi(\cdot|s))\big]\right\}.
\tag{occ-TV}$$

*Proof of inner lemma.* Let $\mu_t^\pi$ be the state distribution at
time $t$ under $\pi$, and define the time-$t$ state--action law
$\rho_t^\pi(s,a):=\mu_t^\pi(s)\pi(a|s)$. Then
$$d^\pi = (1-\gamma)\sum_{t=0}^\infty \gamma^t\rho_t^\pi,
\qquad
d^{\pi'} = (1-\gamma)\sum_{t=0}^\infty \gamma^t\rho_t^{\pi'}.$$ By
convexity of TV under mixtures, $$D_{TV}(d^{\pi'},d^\pi)
\le
(1-\gamma)\sum_{t=0}^\infty \gamma^t D_{TV}(\rho_t^{\pi'},\rho_t^\pi).
\tag{1}$$ Fix $t$. Using $\|\cdot\|_1$ and the identity
$\rho_t^{\pi'}-\rho_t^\pi=(\mu_t^{\pi'}-\mu_t^\pi)\pi' + \mu_t^\pi(\pi'-\pi)$,
$$\|\rho_t^{\pi'}-\rho_t^\pi\|_1
\le
\|\mu_t^{\pi'}-\mu_t^\pi\|_1
+
\|\mu_t^\pi(\pi'-\pi)\|_1.$$ But
$\|\mu_t^\pi(\pi'-\pi)\|_1 = \mathbb{E}_{s\sim \mu_t^\pi}\|\pi'(\cdot|s)-\pi(\cdot|s)\|_1 = 2\mathbb{E}_{s\sim\mu_t^\pi} D_{TV}(\pi'(\cdot|s),\pi(\cdot|s))$.
Therefore $$D_{TV}(\rho_t^{\pi'},\rho_t^\pi)
= \tfrac12\|\rho_t^{\pi'}-\rho_t^\pi\|_1
\le
\tfrac12\|\mu_t^{\pi'}-\mu_t^\pi\|_1
+
\mathbb{E}_{s\sim\mu_t^\pi} D_{TV}(\pi'(\cdot|s),\pi(\cdot|s)).
\tag{2}$$ Now bound the state-distribution difference. Let
$P_\pi(\cdot|s):=\int \pi(a|s)P(\cdot|s,a)\,da$ be the state transition
kernel under $\pi$. Then $$\mu_{t+1}^{\pi'}-\mu_{t+1}^\pi
= \mu_t^{\pi'}P_{\pi'}-\mu_t^\pi P_\pi
= (\mu_t^{\pi'}-\mu_t^\pi)P_{\pi'}
+
\mu_t^\pi(P_{\pi'}-P_\pi).$$ Taking $\|\cdot\|_1$, using that Markov
kernels are non-expansive in $\|\cdot\|_1$,
$$\|\mu_{t+1}^{\pi'}-\mu_{t+1}^\pi\|_1
\le
\|\mu_t^{\pi'}-\mu_t^\pi\|_1
+
\|\mu_t^\pi(P_{\pi'}-P_\pi)\|_1.
\tag{3}$$ For each fixed $s$, $$\|P_{\pi'}(\cdot|s)-P_\pi(\cdot|s)\|_1
= \left\|\int (\pi'(a|s)-\pi(a|s))P(\cdot|s,a)\,da\right\|_1
\le
\int |\pi'(a|s)-\pi(a|s)| \,da
= \|\pi'(\cdot|s)-\pi(\cdot|s)\|_1,$$ hence
$\|P_{\pi'}(\cdot|s)-P_\pi(\cdot|s)\|_1 \le 2D_{TV}(\pi'(\cdot|s),\pi(\cdot|s))$.
Averaging over $s\sim \mu_t^\pi$, $$\|\mu_t^\pi(P_{\pi'}-P_\pi)\|_1
\le
\mathbb{E}_{s\sim \mu_t^\pi}\|P_{\pi'}(\cdot|s)-P_\pi(\cdot|s)\|_1
\le
2\mathbb{E}_{s\sim \mu_t^\pi}D_{TV}(\pi'(\cdot|s),\pi(\cdot|s)).
\tag{4}$$ Let $\delta_t:=\|\mu_t^{\pi'}-\mu_t^\pi\|_1$. Since
$\mu_0^{\pi'}=\mu_0^\pi=\mu$, $\delta_0=0$, and (3)--(4) give
$$\delta_{t+1}\le \delta_t + 2a_t,
\quad\text{where } a_t:=\mathbb{E}_{s\sim\mu_t^\pi}D_{TV}(\pi'(\cdot|s),\pi(\cdot|s)).$$
Thus by induction $\delta_t \le 2\sum_{k=0}^{t-1} a_k$ for $t\ge 1$.
Plug into (2) and then into (1): $$\begin{aligned}
D_{TV}(d^{\pi'},d^\pi)
&\le
(1-\gamma)\sum_{t=0}^\infty \gamma^t\left(\tfrac12\delta_t + a_t\right) \\
&\le
(1-\gamma)\sum_{t=0}^\infty \gamma^t a_t
+
(1-\gamma)\sum_{t=1}^\infty \gamma^t\left(\sum_{k=0}^{t-1} a_k\right).
\end{aligned}$$ The first term equals
$$(1-\gamma)\sum_{t=0}^\infty \gamma^t a_t
= \mathbb{E}_{s\sim d_S^\pi}D_{TV}(\pi'(\cdot|s),\pi(\cdot|s))
=:\overline D_{TV}^{\pi}(\pi',\pi).$$ For the second term, swap sums:
$$(1-\gamma)\sum_{t=1}^\infty \gamma^t\sum_{k=0}^{t-1}a_k
= (1-\gamma)\sum_{k=0}^\infty a_k\sum_{t=k+1}^\infty \gamma^t
= (1-\gamma)\sum_{k=0}^\infty a_k\cdot \frac{\gamma^{k+1}}{1-\gamma}
= \gamma \sum_{k=0}^\infty \gamma^k a_k
= \frac{\gamma}{1-\gamma}\overline D_{TV}^{\pi}(\pi',\pi).$$ Therefore
$$D_{TV}(d^{\pi'},d^\pi)
\le
\left(1+\frac{\gamma}{1-\gamma}\right)\overline D_{TV}^{\pi}(\pi',\pi)
= \frac{1}{1-\gamma}\overline D_{TV}^{\pi}(\pi',\pi).$$ Finally, since
TV is always $\le 1$, we may cap: $$D_{TV}(d^{\pi'},d^\pi)
\le
\min\left\{1,\ \frac{1}{1-\gamma}\overline D_{TV}^{\pi}(\pi',\pi)\right\}.$$
This proves (occ-TV). 0◻

Apply (occ-TV) with $\pi=\pi_t$, $\pi'=\pi_{t+1}$ to get
$$D_{TV}(d_{t+1},d_t)\le \min\left\{1,\ \frac{\overline D_{TV,t}}{1-\gamma}\right\}.$$

**Step 5: Finish the occupancy shift term** From Step 3,
$$\|T_{\mathrm{occ}}\|_2 \le 2M D_{TV}(d_{t+1},d_t)
\le
2M\cdot \min\left\{1,\ \frac{\overline D_{TV,t}}{1-\gamma}\right\}.$$

**Step 6: Combine** Combine Steps 2 and 5 with Step 1 to obtain (L1-TV).
The general divergence form (L1-general) follows immediately by
substituting the assumed per-state bound (TV-compare) inside
$\overline D_{TV,t}$. The KL specialization (L1-KL) follows from Pinsker
plus Jensen: $$\overline D_{TV,t}
= \mathbb{E}_{s\sim d_{S,t}}[TV_s]
\le
\mathbb{E}\Big[\sqrt{\tfrac12 KL_s}\Big]
\le
\sqrt{\tfrac12\ \mathbb{E}[KL_s]}
= \sqrt{\tfrac12\overline D_{KL,t}}.$$ 0◻

#### Proof of critic-error perturbation bound (critic-pert)

Fix $(\theta,\omega)$ and abbreviate $d:=d^{\pi_\theta}$. Write
$$g = g_\star + \delta g.$$ Then $$\begin{aligned}
G(\theta,\omega)-G_\star(\theta)
&= \mathbb{E}_d\big[(g_\star+\delta g)(g_\star+\delta g)^\top - g_\star g_\star^\top\big] \\
&= \mathbb{E}_d[g_\star \delta g^\top]
+
\mathbb{E}_d[\delta g g_\star^\top]
+
\mathbb{E}_d[\delta g \delta g^\top].
\end{aligned}$$ Use $\|\mathbb{E}[X]\|_2\le \mathbb{E}\|X\|_2$ and
$\|uv^\top\|_2=\|u\|_2\|v\|_2$:
$$\|\mathbb{E}_d[g_\star \delta g^\top]\|_2
\le
\mathbb{E}_d[\|g_\star\|_2\|\delta g\|_2]
\le
\sqrt{\mathbb{E}_d\|g_\star\|_2^2}\ \sqrt{\mathbb{E}_d\|\delta g\|_2^2}
= \sigma(\theta)\varepsilon(\theta,\omega),$$ by Cauchy--Schwarz. The
same bound holds for $\mathbb{E}_d[\delta g g_\star^\top]$. Finally,
$$\|\mathbb{E}_d[\delta g \delta g^\top]\|_2
\le
\mathbb{E}_d\|\delta g \delta g^\top\|_2
= \mathbb{E}_d\|\delta g\|_2^2
= \varepsilon(\theta,\omega)^2.$$ Summing yields
$\|G(\theta,\omega)-G_\star(\theta)\|_2 \le 2\sigma\varepsilon+\varepsilon^2$.
0◻

#### How to use this in your pipeline

Lemma 1 (L1-KL) gives you a rigorous $\|\Delta G\|_2$ bound in terms of:

-   parameter drift $(\Delta\theta,\Delta\omega)$,

-   and expected on-distribution KL (plus the TV cap).

(critic-pert) gives you an additive "critic bias/perturbation budget"
that you can include when your Davis--Kahan target is the true-advantage
geometry $G_\star(\theta)$, rather than the algorithmic geometry
$G(\theta,\omega)$.


## Lemma 2

#### Lemma 2 (Geometric Lever: Subspace Sensitivity via Eigengap)

#### Setup (inherited from Lemma 1)

Let
$G(\theta,\omega)=\mathbb{E}_{(s,a)\sim d^{\pi_\theta}}\big[g(s,a;\theta,\omega)g(s,a;\theta,\omega)^\top\big]$
be the population second-moment ("geometry") matrix, with
$$g(s,a;\theta,\omega):=A_\omega(s,a)\nabla_\theta \log \pi_\theta(a\mid s).$$
For iterates $(\theta_t,\omega_t)$, define $G_t:=G(\theta_t,\omega_t)$
and $\Delta G_t:=G_{t+1}-G_t$. Assume A1--A2 from Lemma 1 (one-step
Lipschitz of the integrand in operator norm, and uniform boundedness).

Let the eigenvalues of the symmetric matrix
$G_t\in\mathbb R^{p\times p}$ be ordered
$\lambda_1(G_t)\ge \cdots \ge \lambda_p(G_t)$. Fix
$k\in\{1,\dots,p-1\}$, and let $U_t\in\mathbb R^{p\times k}$ have
orthonormal columns spanning the top-$k$ eigenspace of $G_t$. Define the
(population) eigengap $$\delta_t := \lambda_k(G_t)-\lambda_{k+1}(G_t).$$

Let $\Theta(U_t,U_{t+1})$ denote the diagonal matrix of principal angles
between $\mathrm{col}(U_t)$ and $\mathrm{col}(U_{t+1})$, and let
$\|\sin\Theta(U_t,U_{t+1})\|_F$ be the standard Frobenius
subspace-distance.

#### Statement

Assume $\delta_t>0$. Then $$\boxed{
\|\sin\Theta(U_t,U_{t+1})\|_F
\le
\frac{2\sqrt{k}}{\delta_t}\|\Delta G_t\|_2.
}
\tag{L2-DK}$$ Consequently, combining with Lemma 1 gives the explicit RL
step-to-step rotation bound $$\boxed{
\|\sin\Theta(U_t,U_{t+1})\|_F
\le
\frac{2\sqrt{k}}{\delta_t}\Big(
L_\theta\|\theta_{t+1}-\theta_t\|
+
L_\omega\|\omega_{t+1}-\omega_t\|
+
2M\cdot \min\Big\{1,\frac{D_{\mathrm{TV},t}}{1-\gamma}\Big\}
\Big).
}
\tag{L2-TV}$$ Using the KL instantiation from Lemma 1 (Pinsker +
Jensen), this further implies $$\boxed{
\|\sin\Theta(U_t,U_{t+1})\|_F
\le
\frac{2\sqrt{k}}{\delta_t}\Big(
L_\theta\|\theta_{t+1}-\theta_t\|
+
L_\omega\|\omega_{t+1}-\omega_t\|
+
\min\Big\{2M,\frac{M\sqrt{2}}{1-\gamma}\sqrt{D_{\mathrm{KL},t}}\Big\}
\Big).
}
\tag{L2-KL}$$ All quantities
$(L_\theta,L_\omega,M,D_{\mathrm{TV},t},D_{\mathrm{KL},t})$ are exactly
as defined in Lemma 1.

#### Proof

Apply Theorem 2 (inequality (2)) of Yu--Wang--Samworth (a Davis--Kahan
variant) to the symmetric pair $\Sigma=G_t$ and $\hat\Sigma=G_{t+1}$,
with indices $r=1$ and $s=k$. In that theorem, the "population"
separation in the denominator is
$$\min\big(\lambda_{r-1}(\Sigma)-\lambda_r(\Sigma),\;\lambda_s(\Sigma)-\lambda_{s+1}(\Sigma)\big)
= \min\big(\lambda_0(G_t)-\lambda_1(G_t),\;\lambda_k(G_t)-\lambda_{k+1}(G_t)\big)
= \delta_t,$$ because $\lambda_0:=\infty$. With $d=s-r+1=k$, the theorem
yields $$\|\sin\Theta(U_t,U_{t+1})\|_F
\le
\frac{2\min\big(\sqrt{k}\|\Delta G_t\|_2,\;\|\Delta G_t\|_F\big)}{\delta_t}
\le
\frac{2\sqrt{k}}{\delta_t}\|\Delta G_t\|_2,$$ which proves (L2-DK). Now
invoke Lemma 1's operator-norm perturbation bound for $\|\Delta G_t\|_2$
(either the TV form or the KL specialization) and substitute it into
(L2-DK), yielding (L2-TV) and (L2-KL). 0◻

#### Blow-up / non-identifiability as $\delta_t\to 0$

The bound (L2-DK) shows quantitatively that the subspace map
$G\mapsto\mathrm{col}(U)$ becomes arbitrarily ill-conditioned as
$\delta_t\downarrow 0$: for fixed $\|\Delta G_t\|_2>0$,
$$\|\sin\Theta(U_t,U_{t+1})\|_F \;\le\; \frac{2\sqrt{k}}{\delta_t}\,\|\Delta G_t\|_2 \;=\; O\!\left(\frac{1}{\delta_t}\right).$$
Moreover, this $1/\delta_t$ dependence is sharp in the worst case (there
exist symmetric matrices $G_t$ and perturbations $\Delta G_t$ for which
the resulting subspace rotation is on the order of
$\|\Delta G_t\|_2/\delta_t$ up to constants).

Moreover, when $\delta_t=0$ (i.e. $\lambda_k(G_t)=\lambda_{k+1}(G_t)$),
the "top-$k$" subspace is not uniquely defined, and arbitrarily small
perturbations can select widely different top-$k$ invariant subspaces
(lack of continuity of spectral projectors at eigenvalue collisions).
This is the rigorous sense in which "rank collapse" makes the learning
direction intrinsically unstable.

#### Optional refinement (Moderate-gap regime, Tran--Vu 2025)

Your qualitative discussion distinguishes "diffusion" from "shocks" when
the gap is only a small multiple of the perturbation. In that regime,
classical Davis--Kahan bounds of the form $\|\Delta G\|/\delta$ can be
pessimistic if the perturbation is weakly aligned with the signal
subspace. Tran--Vu (2025) provide a bound that separates (i) a
noise-to-signal ratio $\|\Delta G\|/|\lambda_k|$ and (ii) a correlation
term $x/\delta$. (arXiv)

#### Corollary (Moderate-gap projection bound; spectral norm)

Let $A:=G_t$, $E:=\Delta G_t$, and $\tilde A:=A+E=G_{t+1}$. Let
$\Pi_t:=U_tU_t^\top$ and $\Pi_{t+1}:=U_{t+1}U_{t+1}^\top$ be the
orthogonal projectors onto the top-$k$ eigenspaces of $G_t$ and
$G_{t+1}$. Define $\sigma_1:=\|A\|_2$, and let $r\ge k$ be the smallest
integer such that $|\lambda_k(A)|/2\le |\lambda_k(A)-\lambda_{r+1}(A)|$.
Define the alignment parameter
$$x_t := \max_{i,j\le r} |u_i^\top E u_j|,$$ where $\{u_i\}$ are
eigenvectors of $A=G_t$. Assume the moderate gap condition
$$4\|E\|_2 \le \delta_t \le \frac{|\lambda_k(G_t)|}{4}.$$ Then $$\boxed{
\|\Pi_{t+1}-\Pi_t\|_2
\le
24\Big(
\frac{\|E\|_2}{|\lambda_k(G_t)|}\log\Big(\frac{6\sigma_1}{\delta_t}\Big)
+
\frac{r^2 x_t}{\delta_t}
\Big).
}
\tag{MG}$$ Since $\|\Pi_{t+1}-\Pi_t\|_2=\|\sin\Theta(U_t,U_{t+1})\|_2$,
(MG) is a refined largest-angle control in the moderate-gap regime.
(arXiv)

#### Proof

This is a direct specialization of Tran--Vu (2025), Theorem 2.1, with
$p=k$, $A=G_t$, and $E=\Delta G_t$. (arXiv) 0◻


## Lemma 3

Below I use the notation and setup of Lemma 1 and Lemma 2:
$G_t := \mathbb{E}[g_t g_t^\top]$, $U_t\in\mathbb{R}^{p\times k}$ spans
the top-$k$ eigenspace of $G_t$, and $\Pi_t := U_tU_t^\top$ is the
orthogonal projector onto that subspace. Lemma 1 controls
$\|\Delta G_t\|_2$ in terms of parameter/critic drift and occupancy
shift. Lemma 2 (a Davis--Kahan-type result) controls the *subspace
rotation* (principal angles / projector movement) in terms of
$\|\Delta G_t\|_2$ and the eigengap
$\delta_t := \lambda_k(G_t)-\lambda_{k+1}(G_t)$.

#### Lemma 3 (Shock $\Rightarrow$ update leakage / misalignment)

Let $\Pi_t,\Pi_{t+1}\in\mathbb{R}^{p\times p}$ be orthogonal projectors
(i.e., $\Pi^\top=\Pi$ and $\Pi^2=\Pi$). Let $v\in\mathbb{R}^p$ be any
vector (think: an "update direction", e.g. $v=\Delta\theta_t$). Assume
the update is *mostly in the old signal subspace* in the sense that for
some $\varepsilon\ge 0$, $$\|(I-\Pi_t)v\| \le \varepsilon \|v\|.
\tag{A}$$ Then the "off-subspace" magnitude *with respect to the new
subspace* obeys the two-sided bound
$$\bigl|\|(I-\Pi_{t+1})v\|-\|(\Pi_{t+1}-\Pi_t)v\|\bigr| \le \|(I-\Pi_t)v\|.
\tag{L3-bridge}$$ In particular, under (A),
$$\|(I-\Pi_{t+1})v\| \ge \|(\Pi_{t+1}-\Pi_t)v\| - \varepsilon\|v\|
= \bigl(\alpha_t(v)-\varepsilon\bigr)\|v\|,
\tag{L3-lower}$$ where
$$\alpha_t(v) := \frac{\|(\Pi_{t+1}-\Pi_t)v\|}{\|v\|} \in [0,\ \|\Pi_{t+1}-\Pi_t\|_2]
\quad (v\neq 0).$$

#### Proof (fully self-contained)

Start from the identity $$(I-\Pi_{t+1})v
= v-\Pi_{t+1}v
= (v-\Pi_t v) + (\Pi_t v-\Pi_{t+1}v)
= (I-\Pi_t)v + (\Pi_t-\Pi_{t+1})v.
\tag{1}$$ Take norms and apply the triangle inequality to (1):
$$\|(I-\Pi_{t+1})v\|
\le \|(I-\Pi_t)v\| + \|(\Pi_t-\Pi_{t+1})v\|
= \|(I-\Pi_t)v\| + \|(\Pi_{t+1}-\Pi_t)v\|.
\tag{2}$$ Also apply the **reverse** triangle inequality
$\|x+y\|\ge \bigl|\|y\|-\|x\|\bigr|$ to (1) with $x=(I-\Pi_t)v$ and
$y=(\Pi_t-\Pi_{t+1})v$: $$\|(I-\Pi_{t+1})v\|
\ge \bigl|\|(\Pi_t-\Pi_{t+1})v\|-\|(I-\Pi_t)v\|\bigr|
= \bigl|\|(\Pi_{t+1}-\Pi_t)v\|-\|(I-\Pi_t)v\|\bigr|.
\tag{3}$$ Combining (2) and (3) yields (L3-bridge). Substituting
assumption (A) into the lower side of (3) gives (L3-lower). 0◻

#### Corollary (plug in Lemma 2 to express the "shock" scale via $\|\Delta G_t\|_2/\delta_t$)

In the eigenspace setting of Lemma 2 (so $\Pi_t=U_tU_t^\top$ and
$\Pi_{t+1}=U_{t+1}U_{t+1}^\top$), one always has
$$\|(\Pi_{t+1}-\Pi_t)v\| \le \|\Pi_{t+1}-\Pi_t\|_2 \|v\|.
\tag{4}$$ Moreover, Lemma 2 gives the Frobenius control
$\|\sin\Theta(U_t,U_{t+1})\|_F \le \frac{2\sqrt{k}}{\delta_t}\|\Delta G_t\|_2$.
Using $\|\sin\Theta\|_2 \le \|\sin\Theta\|_F$ and the standard identity
$\|\Pi_{t+1}-\Pi_t\|_2=\|\sin\Theta(U_t,U_{t+1})\|_2$ (both are
immediate from singular-value characterizations of principal angles), we
get $$\|\Pi_{t+1}-\Pi_t\|_2
\le \|\sin\Theta(U_t,U_{t+1})\|_F
\le \frac{2\sqrt{k}}{\delta_t}\|\Delta G_t\|_2.
\tag{5}$$ Defining the risk index $R_t := \|\Delta G_t\|_2/\delta_t$,
(5) reads $$\|\Pi_{t+1}-\Pi_t\|_2 \le 2\sqrt{k} R_t.
\tag{6}$$

**Interpretation consistent with the bridge:** Lemma 3 says the new
off-subspace mass $\|(I-\Pi_{t+1})v\|$ is controlled (up to the old
leakage $\|(I-\Pi_t)v\|$) by the *actual* action of $(\Pi_{t+1}-\Pi_t)$
on $v$. Lemma 2 says $\Pi_{t+1}-\Pi_t$ itself is small whenever
$\|\Delta G_t\|_2/\delta_t$ is small (and can become large when
$\delta_t$ collapses), exactly matching the "shock $\Rightarrow$
staleness/misalignment becomes possible" narrative.

#### Important rigor note about the slide version

The inequality
$$\|(\Pi_{t+1}-\Pi_t)v\| \ge \|\Pi_{t+1}-\Pi_t\|_2 \|v\|$$ is **not true
in general** for a fixed $v$: $\|\Pi_{t+1}-\Pi_t\|_2$ is a *maximum*
over directions, so it only upper-bounds $\|(\Pi_{t+1}-\Pi_t)v\|$. The
rigorous replacement is exactly Lemma 3's $\alpha_t(v)$, i.e. the
*directional* shock seen by your actual update $v$. If you want the
slide's form, you must add an explicit **alignment** assumption (e.g.
$\|(\Pi_{t+1}-\Pi_t)v\| \ge \kappa\|\Pi_{t+1}-\Pi_t\|_2\|v\|$ for some
$\kappa>0$), after which Lemma 3 immediately yields
$\|(I-\Pi_{t+1})v\| \ge (\kappa\|\Pi_{t+1}-\Pi_t\|_2-\varepsilon)\|v\|$.


## Lemma 4

#### Lemma 4 (Misalignment forces a bad certificate via smoothness + signal-subspace structure)

#### Setup

Fix time $t$. Let $F_t:\mathbb{R}^p\to\mathbb{R}$ be differentiable, let
$\theta_{t+1}=\theta_t+\Delta\theta_t$, and let $\Pi_{t+1}$ be an
orthogonal projector (the "current signal subspace" at time $t+1$).

Define:

-   the (true) local gradient $g_t:=\nabla F_t(\theta_t)$,

-   the misalignment of the update against $\Pi_{t+1}$,
    $$m_t := \frac{\|(I-\Pi_{t+1})\Delta\theta_t\|}{\|\Delta\theta_t\|}\in[0,1],
        \qquad(\text{define }m_t:=0\text{ if }\Delta\theta_t=0),$$

-   the smoothness progress certificate
    $$C_t := \langle g_t,\Delta\theta_t\rangle-\frac{L}{2}\|\Delta\theta_t\|^2,$$

-   the bad-step (certificate-collapse) event
    $$\mathsf{Bad}_t := \{C_t\le 0\}
        = \Big\{\langle g_t,\Delta\theta_t\rangle\le \frac{L}{2}\|\Delta\theta_t\|^2\Big\}.$$
    (So $\mathsf{Bad}_t$ means the standard smoothness lower bound does
    not certify improvement; it does not assert
    $F_t(\theta_{t+1})\le F_t(\theta_t)$.)

Also define the realized signal-step ratio $$\rho_t :=
\begin{cases}
\dfrac{\|\Delta\theta_t\|}{\|\Pi_{t+1}g_t\|}, & \text{if }\|\Pi_{t+1}g_t\|>0,\\[8pt]
+\infty, & \text{if }\|\Pi_{t+1}g_t\|=0.
\end{cases}
\tag{$\rho$}$$ (So whenever $\|\Pi_{t+1}g_t\|>0$, we have the identity
$\|\Pi_{t+1}g_t\|=\|\Delta\theta_t\|/\rho_t$.)

#### Assumptions (unchanged except A4.3 removed)

**A4.1 (L-smoothness).** $F_t$ has $L$-Lipschitz gradient, i.e. for all
$\Delta$,
$$F_t(\theta_t+\Delta) \ge F_t(\theta_t)+\langle g_t,\Delta\rangle-\frac{L}{2}\|\Delta\|^2.
\tag{S}$$ **A4.2 (Useful gradient lives in the signal subspace).** There
exists $\beta_t\ge 0$ such that
$$\|(I-\Pi_{t+1})g_t\| \le \beta_t \|\Pi_{t+1}g_t\|.
\tag{G}$$ **Optionally, to connect to Lemma 3:** **A4.4 (Update mostly
in the old subspace $\Pi_t$).** There exists $\varepsilon_t\ge 0$ such
that $$\|(I-\Pi_t)\Delta\theta_t\| \le \varepsilon_t\|\Delta\theta_t\|.
\tag{Old}$$

#### Statement

Let $$c_t := \frac{L\rho_t}{2}\in(0,+\infty],
\qquad
f_{\beta}(m):=\sqrt{1-m^2}+\beta m,\quad m\in[0,1].$$ Define the
high-misalignment threshold $m_\star(\beta,c)$ for $\beta\ge 0$,
$c\in(0,+\infty]$ by $$m_\star(\beta,c):=
\begin{cases}
\text{no threshold exists}, & c<\beta,\\[4pt]
0, & c\ge \sqrt{1+\beta^2},\\[8pt]
\dfrac{\beta c+\sqrt{1+\beta^2-c^2}}{1+\beta^2},
& \beta\le c\le \sqrt{1+\beta^2}.
\end{cases}
\tag{m(*)}$$ ("No threshold exists" means: there is no $m_\star\in[0,1]$
such that $m\ge m_\star\Rightarrow f_\beta(m)\le c$.)

Then:

1.  **(Deterministic bridge: misalignment $\Rightarrow$ bad
    certificate)** If $c_t\ge \beta_t$, then on every outcome where
    $m_t\ge m_\star(\beta_t,c_t)$, one has $\mathsf{Bad}_t$.
    Consequently,
    $$\Pr[\mathsf{Bad}_t] \ge \Pr\big[m_t\ge m_\star(\beta_t,c_t)\big].
        \tag{L4-main}$$ If $c_t<\beta_t$, no implication of the form
    "$m_t\ge m_\star\Rightarrow\mathsf{Bad}_t$" can hold uniformly over
    $m_\star\in[0,1]$ using only the argument below.

2.  **(Plug in Lemma 3: shock/leakage $\Rightarrow$ bad certificate)**
    Under A4.4, Lemma 3 gives
    $$m_t \ge \alpha_t(\Delta\theta_t)-\varepsilon_t,
        \qquad
        \alpha_t(v):=\frac{\|(\Pi_{t+1}-\Pi_t)v\|}{\|v\|},
        \tag{L3}$$ and therefore (when $c_t\ge \beta_t$)
    $$\Pr[\mathsf{Bad}_t]
        \ge
        \Pr\Big[\alpha_t(\Delta\theta_t) \ge m_\star(\beta_t,c_t)+\varepsilon_t\Big].
        \tag{L4-L3}$$

#### Proof

**Step 0: smoothness defines the "certificate"** By A4.1 with
$\Delta=\Delta\theta_t$, $$F_t(\theta_{t+1})-F_t(\theta_t)
\ge
\langle g_t,\Delta\theta_t\rangle-\frac{L}{2}\|\Delta\theta_t\|^2
= C_t.$$ Thus $\mathsf{Bad}_t=\{C_t\le 0\}$ is exactly the event that
the standard smoothness lower bound does not certify improvement.

**Step 1: upper bound the alignment term using subspace components** Let
$$g_\parallel:=\Pi_{t+1}g_t,\quad g_\perp:=(I-\Pi_{t+1})g_t,\qquad
\Delta_\parallel:=\Pi_{t+1}\Delta\theta_t,\quad \Delta_\perp:=(I-\Pi_{t+1})\Delta\theta_t.$$
Orthogonality yields
$$\langle g_t,\Delta\theta_t\rangle=\langle g_\parallel,\Delta_\parallel\rangle+\langle g_\perp,\Delta_\perp\rangle
\le \|g_\parallel\|\|\Delta_\parallel\|+\|g_\perp\|\|\Delta_\perp\|.
\tag{1}$$ Write $\|\Delta_\perp\|=m_t\|\Delta\theta_t\|$ and
$\|\Delta_\parallel\|=\sqrt{1-m_t^2}\|\Delta\theta_t\|$. Under A4.2,
$\|g_\perp\|\le \beta_t\|g_\parallel\|$. Plug into (1):
$$\langle g_t,\Delta\theta_t\rangle
\le
\|g_\parallel\|\|\Delta\theta_t\|\Big(\sqrt{1-m_t^2}+\beta_t m_t\Big)
= \|g_\parallel\|\|\Delta\theta_t\|f_{\beta_t}(m_t).
\tag{2}$$

**Step 2: convert to the bad-certificate inequality using $\rho_t$** If
$\|\Pi_{t+1}g_t\|=0$, then A4.2 implies $\|g_t\|=0$, hence
$$C_t = -\frac{L}{2}\|\Delta\theta_t\|^2\le 0,$$ so $\mathsf{Bad}_t$
holds trivially. Assume henceforth $\|g_\parallel\|=\|\Pi_{t+1}g_t\|>0$,
so $\rho_t<\infty$ and $\|g_\parallel\|=\|\Delta\theta_t\|/\rho_t$ by
definition of $\rho_t$. Substitute into (2):
$$\langle g_t,\Delta\theta_t\rangle
\le
\frac{\|\Delta\theta_t\|^2}{\rho_t}f_{\beta_t}(m_t).$$ Therefore, if
$$f_{\beta_t}(m_t)\le \frac{L\rho_t}{2}=c_t,$$ then
$$\langle g_t,\Delta\theta_t\rangle \le \frac{L}{2}\|\Delta\theta_t\|^2,$$
i.e. $\mathsf{Bad}_t$ occurs. So it remains to show: when
$c_t\ge\beta_t$, the condition $m_t\ge m_\star(\beta_t,c_t)$ implies
$f_{\beta_t}(m_t)\le c_t$.

**Step 3: solve $f_\beta(m)\le c$ and identify the "large-$m$" region**
Fix $\beta\ge 0$ and $c\in(0,\infty]$.

-   If $c<\beta$, then $f_\beta(1)=\beta>c$, so even $m=1$ does not
    satisfy $f_\beta(m)\le c$. Hence no threshold $m_\star\in[0,1]$ can
    ensure $m\ge m_\star\Rightarrow f_\beta(m)\le c$.

-   If $c\ge \sqrt{1+\beta^2}$, then
    $\max_{m\in[0,1]} f_\beta(m)=\sqrt{1+\beta^2}\le c$, so
    $f_\beta(m)\le c$ for all $m$, and we may take $m_\star=0$.

-   If $\beta\le c\le \sqrt{1+\beta^2}$, solve the equality
    $$\sqrt{1-m^2}+\beta m=c.$$ Squaring yields
    $$(1+\beta^2)m^2-2\beta c m+(c^2-1)=0,$$ whose discriminant is
    $4(1+\beta^2-c^2)\ge 0$. The larger root is
    $$m_+(\beta,c)=\frac{\beta c+\sqrt{1+\beta^2-c^2}}{1+\beta^2}.$$
    Moreover, $f_\beta$ increases up to $m=\beta/\sqrt{1+\beta^2}$ and
    decreases thereafter, so for all $m\ge m_+(\beta,c)$ one has
    $f_\beta(m)\le c$.

This matches the definition $m_\star(\beta,c)$ in (m(\*)). Therefore
(when $c_t\ge\beta_t$),
$$m_t\ge m_\star(\beta_t,c_t) \Longrightarrow f_{\beta_t}(m_t)\le c_t
\Longrightarrow \mathsf{Bad}_t,$$ proving the deterministic implication
and hence (L4-main) by taking probabilities.

**Step 4: plug in Lemma 3 under the "old subspace" premise** Under A4.4,
Lemma 3 (with $v=\Delta\theta_t$) gives
$m_t\ge \alpha_t(\Delta\theta_t)-\varepsilon_t$. Hence (when
$c_t\ge\beta_t$):
$$\{\alpha_t(\Delta\theta_t)\ge m_\star(\beta_t,c_t)+\varepsilon_t\}
\subseteq
\{m_t\ge m_\star(\beta_t,c_t)\}
\subseteq
\mathsf{Bad}_t,$$ which implies (L4-L3). 0◻


## Lemma 5

#### Lemma 5 (Small gap $\Rightarrow$ directional shock with constant probability)

#### Setup (inherits Lemmas 2--4 notation)

Let $G_t\in\mathbb{R}^{p\times p}$ be symmetric with eigenvalues
$\lambda_1(G_t)\ge\cdots\ge \lambda_p(G_t)$ and fix an orthonormal
eigenbasis $\{u_{i,t}\}_{i=1}^p$ (arbitrary if eigenvalues have
multiplicity). Fix $k\in\{1,\dots,p-1\}$ and define the (population)
eigengap $$\delta_t := \lambda_k(G_t)-\lambda_{k+1}(G_t) > 0.$$ Let
$\Pi_t$ and $\Pi_{t+1}$ be the orthogonal projectors onto the top-$k$
eigenspaces of $G_t$ and $G_{t+1}$, respectively (as in Lemmas 2--4).

Let $\Delta G_t := G_{t+1}-G_t$ and define the **mixing scalar (boundary
coupling)** $$Z_t := u_{k+1,t}^\top \Delta G_t u_{k,t}.$$ Let
$v_t\neq 0$ be an $\mathcal{F}_t$-measurable update direction (e.g.
$v_t=\Delta\theta_t$), and define the **directional shock** (Lemma 3)
$$\alpha_t(v_t) := \frac{\|(\Pi_{t+1}-\Pi_t)v_t\|}{\|v_t\|}\in[0,1].$$

#### Structural assumptions (the "2$\times$`<!-- -->`{=html}2 reduction" made rigorous)

Let $$\Delta_t^{\uparrow} :=
\begin{cases}
\lambda_{k-1}(G_t)-\lambda_k(G_t), & k\ge 2,\\
+\infty,&k=1,
\end{cases}
\qquad
\Delta_t^{\downarrow} :=
\begin{cases}
\lambda_{k+1}(G_t)-\lambda_{k+2}(G_t), & k+2\le p,\\
+\infty,&k=p-1.
\end{cases}$$

Assume:

**(A5.1) Isolation from neighboring clusters (no cross-ordering with
$\{k,k+1\}$).**
$$\|\Delta G_t\|_2 \le \tfrac14 \min\{\Delta_t^{\uparrow},\Delta_t^{\downarrow}\}.
\tag{Iso}$$

**(A5.2) Two-by-two boundary coupling model (no coupling of
$\{u_{k,t},u_{k+1,t}\}$ to the rest).** For all $i\notin\{k,k+1\}$,
$$u_{i,t}^\top \Delta G_t u_{k,t}=0
\quad\text{and}\quad
u_{i,t}^\top \Delta G_t u_{k+1,t}=0.
\tag{2$\times$2-block}$$

**(A5.3) Update lies in the old signal subspace (as in your pipeline).**
$$v_t \in \mathrm{col}(\Pi_t).
\tag{Old-subspace}$$ Define the boundary-overlap fraction
$$\kappa_t := \frac{|\langle v_t,u_{k,t}\rangle|}{\|v_t\|}\in[0,1].$$

#### Deterministic conclusion (lower bound: mixing forces rotation)

Under (A5.1)--(A5.3),
$$\alpha_t(v_t) \ge \kappa_t \cdot \frac{|Z_t|}{\delta_t + 4\|\Delta G_t\|_2}.
\tag{L5-det}$$ In particular, if $|Z_t|\ge c_Z\|\Delta G_t\|_2$ for some
$c_Z\in(0,1]$, then
$$\alpha_t(v_t) \ge \frac{\kappa_t c_Z}{5} \min\Bigl\{1,\frac{\|\Delta G_t\|_2}{\delta_t}\Bigr\}.
\tag{L5-scale}$$

#### Probabilistic upgrade (constant probability shocks when the gap collapses)

Assume in addition:

**(A5.4) Mixing-entry anti-alignment.** There exist constants
$c_Z\in(0,1]$, $p_0\in(0,1]$ such that
$$\mathbb{P}\left(|Z_t|\ge c_Z\|\Delta G_t\|_2 \,\middle|\, \mathcal{F}_t\right) \ge p_0
\quad\text{a.s.}
\tag{AC}$$

Then, still under (A5.1)--(A5.3), $$\mathbb{P}\left(
\alpha_t(v_t) \ge \frac{\kappa_t c_Z}{5}\min\Bigl\{1,\frac{\|\Delta G_t\|_2}{\delta_t}\Bigr\}
\,\middle|\,\mathcal{F}_t\right) \ge p_0
\quad\text{a.s.}
\tag{L5-prob}$$ Hence on any outcome where
$\delta_t\le \|\Delta G_t\|_2$ (a "collapsed-gap" regime),
$$\mathbb{P}\left(
\alpha_t(v_t) \ge \frac{\kappa_t c_Z}{5}
\,\middle|\,\mathcal{F}_t\right) \ge p_0.
\tag{L5-const}$$

#### Proof of Lemma 5

**Step 1: Under (A5.2), the problem decouples into an
$\mathcal S\oplus\mathcal S^\perp$ block** Work in the eigenbasis of
$G_t$, i.e. the orthonormal basis $\{u_{i,t}\}_{i=1}^p$. By symmetry,
$$G_t = \sum_{i=1}^p \lambda_i(G_t) u_{i,t}u_{i,t}^\top.$$ Define the 2D
"boundary" subspace and its orthogonal complement
$$\mathcal S:=\mathrm{span}\{u_{k,t},u_{k+1,t}\},\qquad \mathcal S^\perp := \mathcal S^{\perp}.$$
Assumption (A5.2) says exactly that $\Delta G_t$ has no entries
connecting $\mathcal S$ to $\mathcal S^\perp$, i.e.
$$\Delta G_t(\mathcal S)\subseteq \mathcal S
\quad\text{and}\quad
\Delta G_t(\mathcal S^\perp)\subseteq \mathcal S^\perp.$$ Since $G_t$ is
diagonal in this basis, it also leaves $\mathcal S$ and
$\mathcal S^\perp$ invariant. Hence $G_{t+1}=G_t+\Delta G_t$ reduces the
orthogonal decomposition
$$\mathbb{R}^p=\mathcal S\oplus \mathcal S^\perp.
\tag{4}$$ *(Optional notation carried from your original writeup: one
may still define
$\mathcal U_+:=\mathrm{span}\{u_{1,t},\dots,u_{k-1,t}\}$ and
$\mathcal U_-:=\mathrm{span}\{u_{k+2,t},\dots,u_{p,t}\}$, so
$\mathcal S^\perp=\mathcal U_+\oplus\mathcal U_-$, but we do **not**
assume $\mathcal U_+$ and $\mathcal U_-$ are separately invariant under
$G_{t+1}$.)*

**Step 2: (A5.1) implies the top-$k$ space uses exactly one direction
from $\mathcal S$** Let $A := G_{t+1}|_{\mathcal S}$ and
$B := G_{t+1}|_{\mathcal S^\perp}$. By (4), $G_{t+1}$ is orthogonally
block-diagonal with blocks $A$ and $B$. Let the eigenvalues of $A$ be
$$\mu_1\ge \mu_2,$$ with corresponding unit top-eigenvector
$\hat u_t\in \mathcal S$ for $\mu_1$. Let the eigenvalues of $B$ be
$$\nu_1\ge \nu_2\ge \cdots \ge \nu_{p-2}.$$ Then the eigenvalues of
$G_{t+1}$ are precisely the multiset
$\{\mu_1,\mu_2\}\cup\{\nu_j\}_{j=1}^{p-2}$, and its top-$k$ eigenspace
is spanned by the eigenvectors corresponding to the $k$ largest values
among these.

We now show, using (A5.1), that:

1.  $\nu_{k-1}>\mu_2$ (for $k\ge 2$), so the top-$k$ space contains **at
    most one** direction from $\mathcal S$; and

2.  $\mu_1>\nu_k$ (for $k\le p-2$), so the top-$k$ space contains **at
    least one** direction from $\mathcal S$, namely $\hat u_t$.

Together, this implies the top-$k$ eigenspace contains **exactly one**
direction from $\mathcal S$, and it is $\mathrm{span}\{\hat u_t\}$.

**(i) Show $\nu_{k-1}>\mu_2$ when $k\ge 2$.** By Courant--Fischer, since
$\mathcal U_+\subseteq \mathcal S^\perp$ has dimension $k-1$,
$$\nu_{k-1}
\ge
\min_{\substack{x\in\mathcal U_+\ \|x\|=1}} x^\top Bx
=
\min_{\substack{x\in\mathcal U_+\ \|x\|=1}} x^\top G_{t+1}x.$$ For
$x\in\mathcal U_+$, we have $x^\top G_tx \ge \lambda_{k-1}(G_t)$ (since
$G_t$ is diagonal with eigenvalues $\ge \lambda_{k-1}$ on
$\mathcal U_+$), and $|x^\top \Delta G_t x|\le \|\Delta G_t\|_2$. Hence
$$\nu_{k-1}\ge \lambda_{k-1}(G_t)-\|\Delta G_t\|_2.
\tag{5}$$ On the other hand, $\mu_2$ is the smaller eigenvalue of
$A=G_{t+1}|_{\mathcal S}=G_t|_{\mathcal S}+\Delta G_t|_{\mathcal S}$.
Since $G_t|_{\mathcal S}$ has eigenvalues $\lambda_k(G_t)$ and
$\lambda_{k+1}(G_t)$ and
$\|\Delta G_t|_{\mathcal S}\|_2\le \|\Delta G_t\|_2$, Weyl implies
$$\mu_2 \le \lambda_{k+1}(G_t)+\|\Delta G_t\|_2 \le \lambda_k(G_t)+\|\Delta G_t\|_2.
\tag{6}$$ Combining (5)--(6), $$\nu_{k-1}-\mu_2
\ge
\bigl(\lambda_{k-1}(G_t)-\lambda_k(G_t)\bigr)-2\|\Delta G_t\|_2
=
\Delta_t^\uparrow -2\|\Delta G_t\|_2.$$ Under (A5.1),
$\|\Delta G_t\|_2\le \Delta_t^\uparrow/4$, so
$\nu_{k-1}-\mu_2\ge \Delta_t^\uparrow/2>0$. Thus $\nu_{k-1}>\mu_2$. (If
$k=1$, this part is vacuous since $k-1=0$.)

**(ii) Show $\mu_1>\nu_k$ when $k\le p-2$.** Again by Courant--Fischer,
since $\mathcal U_-\subseteq \mathcal S^\perp$ has dimension $p-k-1$,
$$\nu_k
\le
\max_{\substack{x\in\mathcal U_-\ \|x\|=1}} x^\top Bx
=
\max_{\substack{x\in\mathcal U_-\ \|x\|=1}} x^\top G_{t+1}x.$$ For
$x\in\mathcal U_-$, $x^\top G_t x \le \lambda_{k+2}(G_t)$, and
$|x^\top\Delta G_t x|\le \|\Delta G_t\|_2$. Hence
$$\nu_k \le \lambda_{k+2}(G_t)+\|\Delta G_t\|_2.
\tag{7}$$ Also, since $\mu_1\ge \mu_2$ and Weyl on the
$\mathcal S$-block yields
$\mu_2\ge \lambda_{k+1}(G_t)-\|\Delta G_t\|_2$, we get
$$\mu_1 \ge \lambda_{k}(G_t)-\|\Delta G_t\|_2.
\tag{8}$$ Combining (7)--(8), $$\mu_1-\nu_k
\ge
\bigl(\lambda_{k+1}(G_t)-\lambda_{k+2}(G_t)\bigr)-2\|\Delta G_t\|_2
=
\Delta_t^\downarrow -2\|\Delta G_t\|_2.$$ Under (A5.1),
$\|\Delta G_t\|_2\le \Delta_t^\downarrow/4$, so
$\mu_1-\nu_k\ge \Delta_t^\downarrow/2>0$. Thus $\mu_1>\nu_k$. (If
$k=p-1$, this part is vacuous since $k+2>p$.)

**Conclusion of Step 2.** From (i)--(ii), the top-$k$ eigenspace of
$G_{t+1}$ consists of:

-   a $(k-1)$-dimensional subspace $W_{t+1}\subseteq \mathcal S^\perp$
    spanned by eigenvectors of $B$ associated with
    $\nu_1,\dots,\nu_{k-1}$, and

-   the 1D subspace $\mathrm{span}\{\hat u_t\}\subseteq\mathcal S$.

Equivalently, there exists an orthogonal projector $P_{t+1}^\perp$ with
$\mathrm{col}(P_{t+1}^\perp)=W_{t+1}\subseteq \mathcal S^\perp$ such
that $$\Pi_{t+1}=P_{t+1}^\perp + \hat u_t\hat u_t^\top.
\tag{9}$$ Likewise, since $\Pi_t$ projects onto
$\mathrm{span}\{u_{1,t},\dots,u_{k,t}\}=\mathcal U_+\oplus \mathrm{span}\{u_{k,t}\}\subseteq \mathcal S^\perp\oplus \mathcal S$,
we may write $$\Pi_t=\Pi_+ + u_{k,t}u_{k,t}^\top,
\tag{10}$$ where $\Pi_+$ is the projector onto
$\mathcal U_+\subseteq\mathcal S^\perp$.

Fix the sign of $\hat u_t$ so that
$\langle \hat u_t,u_{k,t}\rangle\ge 0$, and define
$\theta_t\in[0,\pi/2]$ by
$\cos\theta_t=\langle \hat u_t,u_{k,t}\rangle$, so
$$\sin\theta_t = \|(I-\Pi_{t+1})u_{k,t}\|.
\tag{11}$$

**Step 3: Under (A5.3), the directional shock obeys
$\alpha_t(v_t)\ge \kappa_t\sin\theta_t$** By (A5.3),
$v_t\in\mathrm{col}(\Pi_t)=\mathcal U_+\oplus \mathrm{span}\{u_{k,t}\}$.
Decompose $$v_t = v_{+,t} + c_t u_{k,t},
\qquad v_{+,t}\in\mathcal U_+,\qquad c_t=\langle v_t,u_{k,t}\rangle.$$
Then $|c_t|/\|v_t\|=\kappa_t$.

Using (9)--(10), $$(\Pi_{t+1}-\Pi_t)v_t
=
(\Pi_{t+1}-\Pi_t)v_{+,t} + c_t(\Pi_{t+1}-\Pi_t)u_{k,t}.
\tag{12}$$ We now locate the two terms in orthogonal subspaces:

-   Since $v_{+,t}\in\mathcal U_+\subseteq \mathcal S^\perp$, and both
    $\Pi_{t+1}$ and $\Pi_t$ map $\mathcal S^\perp$ into
    $\mathcal S^\perp$ (because $\Pi_{t+1}$ has the form (9) with
    $P_{t+1}^\perp$ supported on $\mathcal S^\perp$, and $\Pi_t$ has the
    form (10) with $\Pi_+$ supported on $\mathcal S^\perp$), we have
    $$(\Pi_{t+1}-\Pi_t)v_{+,t}\in \mathcal S^\perp.
        \tag{13}$$

-   Since $u_{k,t}\in\mathcal S$, we have $P_{t+1}^\perp u_{k,t}=0$ and
    $\Pi_+ u_{k,t}=0$. Thus from (9)--(10), $$(\Pi_{t+1}-\Pi_t)u_{k,t}
        =
        (\hat u_t\hat u_t^\top - u_{k,t}u_{k,t}^\top)u_{k,t}
        =
        \langle \hat u_t,u_{k,t}\rangle \hat u_t - u_{k,t}
        \in \mathcal S.
        \tag{14}$$

By (13)--(14), the two vectors in (12) are orthogonal (one lies in
$\mathcal S^\perp$, the other in $\mathcal S$). Therefore,
$$\|(\Pi_{t+1}-\Pi_t)v_t\|^2
=
\|(\Pi_{t+1}-\Pi_t)v_{+,t}\|^2
+
c_t^2\|(\Pi_{t+1}-\Pi_t)u_{k,t}\|^2
\ge
c_t^2\|(\Pi_{t+1}-\Pi_t)u_{k,t}\|^2.$$ Taking square roots and dividing
by $\|v_t\|$, $$\alpha_t(v_t)=\frac{\|(\Pi_{t+1}-\Pi_t)v_t\|}{\|v_t\|}
\ge
\frac{|c_t|}{\|v_t\|}\|(\Pi_{t+1}-\Pi_t)u_{k,t}\|
=
\kappa_t\|(\Pi_{t+1}-\Pi_t)u_{k,t}\|.
\tag{$\star$}$$ Finally, since $u_{k,t}\in\mathcal S$ and $\Pi_{t+1}$
acts on $\mathcal S$ as $\hat u_t\hat u_t^\top$, we have
$$(I-\Pi_{t+1})u_{k,t}=u_{k,t}-\langle \hat u_t,u_{k,t}\rangle\hat u_t,$$
whose norm is exactly $\sin\theta_t$. Also
$(\Pi_{t+1}-\Pi_t)u_{k,t}=\Pi_{t+1}u_{k,t}-u_{k,t}=-(I-\Pi_{t+1})u_{k,t}$,
hence $$\|(\Pi_{t+1}-\Pi_t)u_{k,t}\|=\sin\theta_t.$$ Plugging into
($\star$) yields the rigorous bound
$$\alpha_t(v_t)\ge\kappa_t\sin\theta_t.
\tag{$\star\star$}$$

**Step 4: Exact 2$\times$`<!-- -->`{=html}2 eigenvector rotation formula
in $\mathcal S$** Restrict $G_t$ and $G_{t+1}$ to
$\mathcal S=\mathrm{span}\{u_{k,t},u_{k+1,t}\}$ and use the basis
$(u_{k,t},u_{k+1,t})$. In that basis, $$G_t|_{\mathcal S}=
\begin{pmatrix}
\lambda_k(G_t) & 0\\
0 & \lambda_{k+1}(G_t)
\end{pmatrix}.$$ Write the corresponding $2\times 2$ block of
$\Delta G_t$ as $$\Delta G_t|_{\mathcal S}=
\begin{pmatrix}
a_t & Z_t\\
Z_t & b_t
\end{pmatrix},
\quad
a_t:=u_{k,t}^\top\Delta G_t u_{k,t},\;
b_t:=u_{k+1,t}^\top\Delta G_t u_{k+1,t}.$$ So $$G_{t+1}|_{\mathcal S}=
\begin{pmatrix}
\lambda_k+a_t & Z_t\\
Z_t & \lambda_{k+1}+b_t
\end{pmatrix}.$$ If $Z_t=0$, then the desired bound (L5-det) is
immediate since its right-hand side is 0. Assume henceforth $Z_t\neq 0$,
so the top eigenvector in $\mathcal S$ is well-defined up to sign.

For a real symmetric $2\times 2$ matrix
$\begin{psmallmatrix}x & z\\ z & y\end{psmallmatrix}$, the
top-eigenvector makes an angle $\theta$ with the first coordinate axis
satisfying $$\tan(2\theta)=\frac{2z}{x-y}.$$ Here
$x-y = (\lambda_k+a_t)-(\lambda_{k+1}+b_t)=\delta_t + (a_t-b_t)$. Thus
$$\sin(2\theta_t)=\frac{2|Z_t|}{\sqrt{(\delta_t+a_t-b_t)^2+4Z_t^2}}.$$
Since $\sin(2\theta)=2\sin\theta\cos\theta\le 2\sin\theta$ for
$\theta\in[0,\pi/2]$, we get the deterministic lower bound
$$\sin\theta_t \ge \frac{|\sin(2\theta_t)|}{2}
=
\frac{|Z_t|}{\sqrt{(\delta_t+a_t-b_t)^2+4Z_t^2}}.
\tag{1}$$ Next, $\sqrt{u^2+v^2}\le |u|+|v|$, so
$$\sqrt{(\delta_t+a_t-b_t)^2+4Z_t^2}
\le
|\delta_t+a_t-b_t| + 2|Z_t|
\le
\delta_t + |a_t-b_t| + 2|Z_t|.
\tag{2}$$ Finally, for any symmetric matrix $E$,
$|u^\top Eu|\le \|E\|_2$ for unit $u$, hence
$|a_t|\le \|\Delta G_t\|_2$, $|b_t|\le \|\Delta G_t\|_2$, so
$|a_t-b_t|\le 2\|\Delta G_t\|_2$, and also $|Z_t|\le \|\Delta G_t\|_2$.
Combining (1)--(2) gives $$\sin\theta_t
\ge
\frac{|Z_t|}{\delta_t + 2\|\Delta G_t\|_2 + 2|Z_t|}
\ge
\frac{|Z_t|}{\delta_t + 4\|\Delta G_t\|_2}.
\tag{3}$$

**Step 5: Conclude (L5-det), then (L5-scale)** Plug (3) into
($\star\star$):
$$\alpha_t(v_t)\ge\kappa_t\sin\theta_t \ge \kappa_t \frac{|Z_t|}{\delta_t+4\|\Delta G_t\|_2},$$
which is (L5-det).

If additionally $|Z_t|\ge c_Z\|\Delta G_t\|_2$, then
$$\alpha_t(v_t)\ge \kappa_t \frac{c_Z\|\Delta G_t\|_2}{\delta_t+4\|\Delta G_t\|_2}.$$
If $\delta_t\ge \|\Delta G_t\|_2$, then
$\delta_t+4\|\Delta G_t\|_2\le 5\delta_t$, yielding
$\alpha_t(v_t)\ge (\kappa_t c_Z/5)\cdot (\|\Delta G_t\|_2/\delta_t)$. If
$\delta_t<\|\Delta G_t\|_2$, then
$\delta_t+4\|\Delta G_t\|_2\le 5\|\Delta G_t\|_2$, yielding
$\alpha_t(v_t)\ge \kappa_t c_Z/5$. Together this is exactly (L5-scale).

**Step 6: Probabilistic lower bound (L5-prob), and "gap collapse
$\Rightarrow$ constant-probability shock"** Under (A5.4), on the event
$\{|Z_t|\ge c_Z\|\Delta G_t\|_2\}$, inequality (L5-scale) holds
deterministically. Therefore $$\mathbb{P}\left(
\alpha_t(v_t)\ge \frac{\kappa_t c_Z}{5}\min\Bigl\{1,\frac{\|\Delta G_t\|_2}{\delta_t}\Bigr\}
\,\middle|\,\mathcal{F}_t\right)
\ge
\mathbb{P}\left(|Z_t|\ge c_Z\|\Delta G_t\|_2\,\middle|\,\mathcal{F}_t\right)
\ge p_0,$$ which is (L5-prob). The specialization (L5-const) follows
immediately on $\{\delta_t\le \|\Delta G_t\|_2\}$. 0◻
