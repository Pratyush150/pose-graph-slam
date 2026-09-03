# Benchmark datasets: what they are, where they come from, and why none are committed

## Why nothing is committed

These are the standard public pose-graph benchmarks. They are not our data.
The mirrors we fetch them from are licensed LGPL-3.0 (SE-Sync), GPL-3.0
(Vertigo) and BSD-style (GTSAM), and redistributing them inside an
MIT-licensed package would be sloppy at best. So `data/` is empty in git and
`tools/fetch_datasets.py` downloads on demand.

That would normally make results unreproducible, so the registry in
`src/posegraph/datasets.py` records the exact URL, byte size and SHA-256 of
every file the benchmarks in this repository were actually run against. The
fetch script verifies the checksum after every download and refuses a file
that does not match, which catches a mirror that has quietly changed.

```bash
python3 tools/fetch_datasets.py --list      # what is registered, what is on disk
python3 tools/fetch_datasets.py --all       # everything, about 30 MB
python3 tools/fetch_datasets.py intel sphere2500 manhattan manhattan_gt
python3 tools/fetch_datasets.py --verify    # checksum what is already there
```

## Pose graphs

| name | space | size | SHA-256 (first 16) | origin |
|---|---|---:|---|---|
| `intel` | SE2 | 300 KB | `3e0724c048e0ba52` | Intel Research Lab, Seattle (Dirk Haehnel); g2o conversion via SE-Sync |
| `MIT` | SE2 | 119 KB | `e5922be0d0689c7a` | MIT Killian Court (Mike Bosse); g2o conversion via SE-Sync |
| `CSAIL` | SE2 | 125 KB | `66d99ac857a9849d` | MIT CSAIL building; g2o conversion via SE-Sync |
| `manhattan` | SE2 | 581 KB | `6ae8d30971720c1a` | Manhattan3500 / M3500 (Ed Olson); g2o conversion via SE-Sync |
| `city10000` | SE2 | 1.7 MB | `df5988994339e990` | City10000 simulated urban sequence; g2o conversion via SE-Sync |
| `ais2klinik` | SE2 | 2.1 MB | `e388ebd8523987c1` | AIS2 Klinik, University of Freiburg; g2o conversion via SE-Sync |
| `sphere2500` | SE3 | 1.0 MB | `104ab57593394f24` | sphere2500 simulated 3D sphere (Michael Kaess); via SE-Sync |
| `torus3D` | SE3 | 3.1 MB | `1ef8831023b37218` | torus3D simulated 3D torus; via SE-Sync |
| `parking-garage` | SE3 | 1.2 MB | `3ac0a31bfb601d74` | Multi-level parking garage, real vehicle data; via SE-Sync |
| `cubicle` | SE3 | 2.6 MB | `f7781d485383cec8` | Cubicle office environment; via SE-Sync |
| `grid3D` | SE3 | 7.3 MB | `f5dd7a17bf8b9c39` | Simulated 3D grid; via SE-Sync |
| `rim` | SE3 | 4.6 MB | `9cc9b0d8e5f080c5` | RIM dataset; via SE-Sync |
| `smallGrid3D` | SE3 | 99 KB | `9ea56c2ad1ebcc32` | Small simulated 3D grid; via SE-Sync |
| `tinyGrid3D` | SE3 | 4 KB | `c341eb0d09f7556b` | Tiny simulated 3D grid; via SE-Sync |
| `sphere_bignoise` | SE3 | 2.1 MB | `484aa1999084d353` | sphere2500 with inflated measurement noise; via SE-Sync |

Contents, as parsed by this package:

| name | poses | edges | loop-closure edges | vertices in file | note |
|---|---:|---:|---:|---|---|
| `intel` | 1728 | 2512 | 785 | yes | The classic indoor 2D loop-closure test. |
| `MIT` | 808 | 827 | 20 | yes | Very sparse loop closures and long open corridors. |
| `CSAIL` | 1045 | 1172 | 128 | **no** | Indoor 2D office loop. |
| `manhattan` | 3500 | 5453 | 1954 | **no** | Ships edges only, so the initial guess must be chained from odometry. |
| `city10000` | 10000 | 20687 | 10688 | yes | Simulated city grid, the densest 2D loop structure here. |
| `ais2klinik` | 15115 | 16727 | 1614 | yes | The largest 2D graph here; long corridors, few loops. |
| `sphere2500` | 2500 | 4949 | 2450 | yes | The standard SE(3) stress test: large rotations everywhere. |
| `torus3D` | 5000 | 9048 | 4049 | yes | Closes in three dimensions at once; slow to converge. |
| `parking-garage` | 1661 | 6275 | 4615 | yes | Real vehicle data, many short-range constraints. |
| `cubicle` | 5750 | 16869 | 7621 | yes | Densest SE(3) loop-closure structure here. |
| `grid3D` | 8000 | 22236 | 14237 | yes | Simulated 3D grid. |
| `rim` | 10195 | 29743 | 13475 | yes | Large 3D graph. |
| `smallGrid3D` | 125 | 297 | 173 | yes | Fast smoke test for the SE(3) path. |
| `tinyGrid3D` | 9 | 11 | 3 | yes | Nine poses; small enough to inspect by hand. |
| `sphere_bignoise` | 2200 | 8647 | 6448 | yes | Inflated measurement noise; a much harder initial guess than sphere2500. |

## Ground truth

Only three of these datasets have ground truth we could obtain. Absolute
Trajectory Error is reported for those three and left blank everywhere else,
rather than substituting something that is not ground truth.

| name | for | format | size | SHA-256 (first 16) | origin | licence |
|---|---|---|---:|---|---|---|
| `manhattan_gt` | `manhattan` | xytheta | 87 KB | `8512b270849f2571` | Vertigo (Niko Suenderhauf), ground truth for Olson's M3500 | GPL-3.0 |
| `city10000_gt` | `city10000` | xytheta | 211 KB | `a300f29b446608ea` | GTSAM example data | BSD-style (see GTSAM LICENSE) |
| `sphere2500_gt` | `sphere2500` | edge3_chain | 618 KB | `b9cfd29c951586bf` | GTSAM example data (noise-free EDGE3 records) | BSD-style (see GTSAM LICENSE) |

`xytheta` files are one `x y theta` line per pose, in graph order.

`edge3_chain` is the awkward one. GTSAM's `sphere2500_groundtruth.txt` does not
contain poses at all: it contains the same 4949 edges as `sphere2500.g2o` with
the noise removed. The absolute poses come from chaining the 2499 sequential
records from the identity, with the rotation part read as Z-Y-X Euler angles.
The remaining 2450 records are loop closures that were not used to build the
chain, so they are an independent consistency check on it --
`tests/test_datasets.py::test_sphere_ground_truth_is_internally_consistent`
verifies they agree to better than 1e-3 m, which is what makes the file usable
as truth rather than as just another trajectory.

## Awkward details in the files themselves

* **`manhattan.g2o` and `CSAIL.g2o` contain no vertex records at all** -- 5453
  and 1172 `EDGE_SE2` lines respectively, and nothing else. The reader creates
  the missing poses and initialises them by walking a breadth-first spanning
  tree, preferring short-index edges so it follows odometry first. That is what
  produces the drifting initial guess in the before/after figures, and it is
  also why the initial chi-squared for those two files depends on the reader:
  a different spanning tree is a different initial guess.
* Information matrices are stored as an upper triangle in row-major order: 6
  values for `EDGE_SE2`, 21 for `EDGE_SE3:QUAT`, ordered translation first and
  then rotation.
* Quaternions in `VERTEX_SE3:QUAT` and `EDGE_SE3:QUAT` are scalar-last
  (`qx qy qz qw`) and are renormalised on read.
* `sphere_bignoise_vertex3.g2o` has 2200 vertices, not the 2500 its name
  suggests; the registry note records what the file actually contains.
* g2o's own `EDGE_SE3` error definition differs slightly from the one used
  here, so an information matrix taken from a `.g2o` file is being reused under
  a marginally different parameterisation. This is standard practice for a
  from-scratch implementation, and it is why chi-squared values are comparable
  across solvers only up to that convention -- see the cross-check section of
  the README, where the SE(3) initial costs match GTSAM exactly and the SE(2)
  ones do not.
