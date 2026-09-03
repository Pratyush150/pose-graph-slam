# The maths, for an engineer who has not done nonlinear optimisation

This is the whole method, in the order the code performs it. Nothing here needs
more than linear algebra and a willingness to accept one idea: you cannot add
rotations, so you have to be careful about what "add a small correction" means.

## 1. The problem

A front end gives you *relative* measurements. "From pose 41 I moved 1.03 m
forward and turned 4 degrees." "Pose 900 is the same place as pose 12, and the
transform between them is this." Each measurement is a rigid transform with a
noise model.

Call the unknown poses `x_1 ... x_n` and a measurement between poses `i` and `j`
`z_ij`, with information matrix `Omega_ij` (the inverse covariance -- a large
entry means "I am confident about this component").

The *prediction* of that measurement, given a current estimate, is
`x_i^-1 x_j`. The *error* is how far the prediction is from the measurement:

```
e_ij(x) = Log( z_ij^-1 * x_i^-1 * x_j )
```

`Log` maps a rigid transform to a vector: 3 numbers for SE(2)
(`dx, dy, dtheta`), 6 for SE(3) (`dx, dy, dz` and a rotation vector). When the
estimate is perfect, `z_ij^-1 x_i^-1 x_j` is the identity and `e_ij` is zero.

The estimate is the one that makes all the errors small at once, weighted by how
much you trust each:

```
x* = argmin  sum_ij  e_ij(x)^T Omega_ij e_ij(x)
```

That sum is the **chi-squared**. Under a correct Gaussian noise model it should
end up near the number of degrees of freedom (measurements minus unknowns), which
is why `posegraph.analysis` reports `chi2/dof` -- a value near 1 says the noise
model was honest, a value of 50 says something is wrong with the data or the
covariances.

## 2. Why you cannot just take a derivative

`x_i` is a rigid transform, not a vector. There is no meaningful "`x_i` plus
0.01" because rotations do not add: two 90-degree rotations about different axes
do not compose into a 180-degree rotation about their sum.

The fix is to parameterise *corrections* rather than poses. Every rigid transform
near the identity can be written `Exp(d)` for a small vector `d`, and `Exp` is
the matrix exponential of the corresponding twist. So define the retraction

```
x [+] d  =  x * Exp(d)          (right perturbation, used throughout)
```

Now `d` is an ordinary vector living in the tangent space, and the cost is an
ordinary function of `d`. Everything from a first-year optimisation course
applies again -- as long as every derivative is taken with respect to `d`, and
every update is applied through `[+]` rather than `+`.

This is what `posegraph/se2.py` and `posegraph/se3.py` implement: `exp`, `log`,
the adjoint, and the derivatives below.

## 3. The analytic Jacobians

Linearise the error around the current estimate:

```
e_ij(x_i [+] d_i, x_j [+] d_j)  ~  e_ij + J_i d_i + J_j d_j
```

With the right-perturbation convention the two blocks have closed forms. Write
`e = e_ij` and `E = Exp(e)`:

```
J_j =  Jr^-1(e)
J_i = -Jr^-1(e) * Ad( x_j^-1 x_i )
```

`Ad(T)` is the adjoint of `T`: the matrix that moves a tangent vector from one
frame to another, `T Exp(d) T^-1 = Exp(Ad(T) d)`. `Jr(e)` is the right Jacobian
of the group -- the derivative of `Exp` at `e`, which corrects for the fact that
`Exp(a + b)` is not `Exp(a) Exp(b)`.

Deriving `J_i`: perturbing `x_i` gives
`E' = z^-1 Exp(-d_i) x_i^-1 x_j = Exp(-Ad(z^-1) d_i) E`, and
`Log(Exp(u) Exp(e)) ~ e + Jl^-1(e) u`, so `de/dd_i = -Jl^-1(e) Ad(z^-1)`.
Using `Jl^-1(e) = Jr^-1(e) Ad(E^-1)` and `z E = x_i^-1 x_j` collapses that to the
expression above.

For SO(3) the right Jacobian has the standard form

```
Jl(phi) = I + ((1 - cos t)/t^2) [phi]x + ((t - sin t)/t^3) [phi]x^2,   t = |phi|
Jr(phi) = Jl(-phi)
```

For SE(3) this package uses the closed form of the series
`Jl = sum_n ad^n / (n+1)!`. The little adjoint of `se(3)` satisfies
`ad^5 = -2 t^2 ad^3 - t^4 ad`, so the infinite series collapses to a quartic:

```
Jl = I + a ad + b ad^2 + c ad^3 + d ad^4
a = (4 - t sin t - 4 cos t) / (2 t^2)
b = (4 t - 5 sin t + t cos t) / (2 t^3)
c = (2 - t sin t - 2 cos t) / (2 t^4)
d = (2 t - 3 sin t + t cos t) / (2 t^5)
```

That is both shorter and easier to get right than the more common
block form with Barfoot's `Q` matrix, and it reduces to the SO(3) expression
above when the translation part is zero -- which is asserted in
`tests/test_se3.py`.

### The small-angle problem

Every one of those coefficients is `0/0` at `t = 0` and loses precision near it:
`(1 - cos t)/t^2` at `t = 1e-8` computes `1e-16 / 1e-16` in double precision and
returns noise. Each coefficient therefore switches to its Taylor series below a
threshold chosen so *both* branches are accurate there. `d` divides by `t^5` and
so needs a much larger threshold (0.1) than the SO(3) coefficients (0.01).

`Jl^-1` for SO(3) is written with `cot(t/2)` rather than the algebraically equal
`(1 + cos t)/(2 t sin t)`, because the second form is `0/0` at `t = pi` -- exactly
where a near-180-degree loop closure lives.

`tests/test_small_angle_branches.py` asserts the branches meet, and
`tests/test_jacobians.py` checks every analytic Jacobian against central
differences. That second test is the one that makes any of this trustworthy.

## 4. The normal equations

Stack the linearised errors and minimise. The result is the familiar

```
H dx = b
H = sum_ij  J^T Omega J        (block sparse: only blocks (i,i), (i,j), (j,j))
b = -sum_ij J^T Omega e
```

`H` is the *information matrix*. It has one block row and column per pose, and a
nonzero off-diagonal block exactly where an edge exists. A pose graph with 5000
poses and 9000 edges gives a 30000 x 30000 matrix with about 0.01% of its entries
nonzero. Forming it densely is not an option.

### Gauge freedom

`H` is always singular before you do something about it. Rotate and translate the
entire solution rigidly and every *relative* measurement is unchanged, so the
cost is unchanged: there is a 3-dimensional (SE(2)) or 6-dimensional (SE(3))
null space. Fixing one pose -- deleting its rows and columns -- removes it
exactly. `tests/test_solver.py` counts the null-space dimension before and after
to confirm this.

## 5. Solving, without drowning in fill-in

Cholesky factorisation of a sparse matrix creates new nonzeros ("fill-in") that
were not in the original. How many depends entirely on the elimination order. A
trajectory that revisits its start looks like a long chain with one extra edge
between the ends; eliminate it in index order and the factor fills in almost
completely.

`posegraph/linalg.py` computes the fill for three orderings and picks the best:

* **natural** -- the order the file happened to use;
* **reverse Cuthill-McKee** -- breadth-first from a pseudo-peripheral node, then
  reversed, which narrows the bandwidth;
* **minimum degree** -- repeatedly eliminate the vertex with the fewest
  neighbours and make its neighbourhood a clique.

The counts are not estimates. `symbolic_cholesky` runs the row-subtree algorithm
and returns the exact number of nonzero blocks the factor will have, and
`tests/test_ordering.py` checks that number against an explicit dense
elimination on the same pattern.

## 6. Gauss-Newton, Levenberg-Marquardt, dogleg

**Gauss-Newton** solves `H dx = b` and applies `x <- x [+] dx`. It converges
quadratically near the answer and diverges enthusiastically when the initial
guess is bad -- which is the normal case for a pose graph built from raw
odometry.

**Levenberg-Marquardt** solves `(H + lambda D) dx = b` instead. Large `lambda`
gives a short step along the gradient (safe, slow); small `lambda` recovers
Gauss-Newton (fast, risky). After each step the *gain ratio*

```
rho = (actual cost reduction) / (reduction the quadratic model predicted)
```

decides what happens next: `rho > 0` accepts the step and scales `lambda` by
`max(1/3, 1 - (2 rho - 1)^3)`; otherwise the step is undone exactly and `lambda`
is multiplied by a doubling factor. Because rejected steps are undone, the cost
sequence is monotonically non-increasing by construction -- asserted in
`tests/test_solver.py`.

**Dogleg** replaces the damping term with an explicit trust-region radius and
blends the Gauss-Newton step with the Cauchy (steepest-descent) step to stay
inside it. One factorisation per iteration rather than one per inner retry.

## 7. Robust kernels

A single false loop closure -- place recognition matching two corridors that look
alike -- is a confident measurement that is simply wrong. Under a squared cost it
has unbounded influence, and the optimiser will happily fold the map in half to
reduce it.

A kernel replaces `s` (the squared Mahalanobis error of one edge) with `rho(s)`
that grows more slowly. In IRLS form this is one line: scale the edge's
information matrix by `w(s) = drho/ds`.

| kernel | `rho(s)` | `w(s)` | behaviour |
|---|---|---|---|
| Huber | `s`, then `2 delta sqrt(s) - delta^2` | `1`, then `delta/sqrt(s)` | convex, no new minima, safe default |
| Cauchy | `c^2 log(1 + s/c^2)` | `1/(1 + s/c^2)` | suppresses, never deletes |
| Geman-McClure | `c^2 s/(c^2 + s)` | `c^4/(c^2 + s)^2` | strongly redescending, needs a decent start |
| DCS | `s`, then `3 Phi - 4 Phi^2/(Phi + s)` | `min(1, 2 Phi/(Phi + s))^2` | full weight below `Phi`, fast cut-off above |

A detail worth stating because it is easy to get wrong: the DCS paper motivates
the method as scaling the information by `sigma^2`, which suggests a cost of
`sigma^2 s`. That is *not* the antiderivative of `sigma^2`, so using it would
make Levenberg-Marquardt compare an actual reduction in one function against a
predicted reduction in another and produce a wrong gain ratio. This
implementation integrates the weight instead, which is continuous at `Phi`,
saturates at `3 Phi`, and is what the weights are consistent with.

### Choosing the width

`delta` is in units of `sqrt(chi2)`, so it should come from the noise model, not
from habit. A correctly-modelled SE(2) edge has three degrees of freedom and
therefore a chi-squared of about 3 on average; an SE(3) edge about 6. Setting
`delta = 1` -- a very common default -- declares an entirely ordinary residual an
outlier, down-weights half the graph, and biases the solution.

`robust.default_delta(dof)` returns the square root of the chi-squared critical
value at a stated confidence: 2.80 for SE(2) at 95%, 3.55 for SE(3). That is
what the solver uses when no width is given.

The second-order Triggs correction to the Gauss-Newton Hessian is deliberately
not applied: it can make `H` indefinite for redescending kernels, and the LM
damping already covers the same ground.

## 8. What is left out

Chordal or spanning-tree rotation initialisation, certifiably optimal solvers
(SE-Sync and friends solve a semidefinite relaxation and can prove global
optimality), iSAM2-style incremental factorisation with a Bayes tree, and
switchable constraints. Section "What this is not" in the README says where
those would go.

## References

* Grisetti, Kummerle, Stachniss, Burgard. *A Tutorial on Graph-Based SLAM.*
  IEEE Intelligent Transportation Systems Magazine, 2010. The clearest
  introduction to everything in sections 1-4.
* Kummerle, Grisetti, Strasdat, Konolige, Burgard. *g2o: A General Framework for
  Graph Optimization.* ICRA 2011. The reference implementation and the origin of
  the `.g2o` file format.
* Dellaert, Kaess. *Factor Graphs for Robot Perception.* Foundations and Trends
  in Robotics, 2017. The factor-graph view, elimination orderings and
  incremental solving.
* Barfoot. *State Estimation for Robotics.* Cambridge, 2017. Chapter 7 for the
  SE(3) Jacobians used here.
* Solà, Deray, Atchuthan. *A micro Lie theory for state estimation in robotics.*
  arXiv:1812.01537, 2018. Short and practical on conventions.
* Agarwal, Tipaldi, Spinello, Stachniss, Burgard. *Robust Map Optimization using
  Dynamic Covariance Scaling.* ICRA 2013.
* Sunderhauf, Protzel. *Switchable Constraints for Robust Pose Graph SLAM.*
  IROS 2012. The source of the outlier-injected benchmark variants.
* Umeyama. *Least-squares estimation of transformation parameters between two
  point patterns.* IEEE PAMI, 1991. The alignment used by ATE.
