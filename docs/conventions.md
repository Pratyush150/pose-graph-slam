# Conventions

Every sign error in a SLAM back end comes from mixing two of these up. They are
stated once, here, and used consistently everywhere in the package.

## Storage

| thing | layout |
|---|---|
| SE(2) pose | `(x, y, theta)`, shape `(..., 3)` |
| SE(3) pose | `(x, y, z, qx, qy, qz, qw)`, shape `(..., 7)` |
| SE(2) tangent vector | `(dx, dy, dtheta)` |
| SE(3) tangent vector | `(dx, dy, dz, wx, wy, wz)` -- **translation first** |
| 2D point | `(x, y)` |
| 3D point | `(x, y, z)` |

Translation-first ordering is not arbitrary: it is what `.g2o` uses for the
upper-triangular information matrix of `EDGE_SE3:QUAT`, so an information matrix
read from a file drops straight in without a permutation.

Quaternions are Hamilton, scalar-last, unit norm, with a non-negative scalar
part enforced. That last part matters: `q` and `-q` are the same rotation, and
forcing `qw >= 0` makes the logarithm single-valued. They are renormalised after
every product, so a 5000-step composition chain does not drift off the unit
sphere (asserted in `tests/test_se3.py`).

## Perturbation

**Right perturbation**, everywhere:

```
X [+] d  =  X * Exp(d)
X [-] Y  =  Log(Y^-1 * X)
```

`d` is expressed in the *body* frame of `X`. The alternative (left perturbation,
`Exp(d) * X`, with `d` in the world frame) is equally valid and appears in plenty
of literature; mixing the two silently transposes an adjoint and the optimiser
takes steps in the wrong direction while still reducing the cost slightly, which
is a miserable bug to find.

## Measurement and error

`z_ij` is the pose of `j` expressed in the frame of `i`. The error is

```
e_ij = Log( z_ij^-1 * x_i^-1 * x_j )
```

so a perfect estimate gives zero, and the Jacobians are

```
de/dd_j =  Jr^-1(e)
de/dd_i = -Jr^-1(e) * Ad(x_j^-1 x_i)
```

A unary prior on `x` towards `z` uses `e = Log(z^-1 x)` and `de/dd = Jr^-1(e)`.

A pose-to-point measurement is the point in the pose's own frame,
`e = R_i^T (p_j - t_i) - z`, with

```
de/dd_i = [ -I,  [q]x ]   (SE(3), q = R_i^T (p_j - t_i))
de/dp_j = R_i^T
```

## Information, not covariance

Edges carry `Omega = Sigma^-1`. Larger means more confident. The cost of one
edge is `e^T Omega e`, which is dimensionless, so chi-squared values from
different edge types are directly comparable and `chi2/dof` is meaningful.

## Normal equations

```
H dx = b        H = sum J^T (w Omega) J        b = -sum J^T (w Omega) e
```

Note the sign: `b` is the *negative* gradient direction (up to a factor of two),
so the update is `x <- x [+] dx` with no extra minus sign. `w` is the robust
kernel's IRLS weight, `1` when no kernel is in use.

## Angles

`normalize_angle` wraps to `(-pi, pi]`. The SE(2) logarithm always returns an
angle in that range, so `log(exp(xi))` round-trips only for `|theta| < pi` --
which is a property of the group, not a limitation of the implementation.

## Batching

Every function in `se2.py` and `se3.py` accepts either a single element (shape
`(3,)`, `(7,)`, `(6,)`) or a stack (shape `(N, 3)` and so on), and preserves
which it was given. The solver relies on this: linearising 16869 edges in a
Python loop is not viable, so it linearises all of them in one vectorised call.
