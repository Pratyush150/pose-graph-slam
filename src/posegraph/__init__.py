"""posegraph -- a pose-graph SLAM back end written from scratch on NumPy.

Front ends produce noisy relative-pose measurements. This package turns those
measurements into a single globally consistent trajectory by solving a sparse
nonlinear least-squares problem on the SE(2)/SE(3) manifold.

Quick start::

    from posegraph.graph import read_g2o
    from posegraph.solver import optimize, SolverOptions

    g = read_g2o("intel.g2o")
    g.fix_pose(g.pose_ids[0])          # remove the gauge freedom
    result = optimize(g, SolverOptions(method="lm", kernel="huber"))
    print(result.summary())

The submodules are deliberately separable: :mod:`posegraph.se2` and
:mod:`posegraph.se3` are standalone Lie-group libraries, :mod:`posegraph.linalg`
is a standalone sparse-ordering/solve layer, and :mod:`posegraph.solver` glues
them together.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Pratyush Vatsa"

from . import analysis, datasets, frontend_stub, graph, incremental, linalg
from . import robust, se2, se3, solver
from .graph import PoseGraph, read_g2o, write_g2o
from .robust import make_kernel
from .solver import SolverOptions, SolverResult, optimize

__all__ = [
    "__version__",
    "PoseGraph",
    "SolverOptions",
    "SolverResult",
    "analysis",
    "datasets",
    "frontend_stub",
    "graph",
    "incremental",
    "linalg",
    "make_kernel",
    "optimize",
    "read_g2o",
    "robust",
    "se2",
    "se3",
    "solver",
    "write_g2o",
]
