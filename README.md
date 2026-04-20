# GPDF-ADAS

A Python library for representing obstacle boundaries as **Gaussian Process Distance Fields (GPDF)** — a continuous, differentiable signed-distance representation suited for motion planning and ADAS applications.

Originally developed by Ho Jin Choi, Yifan Xue, and Nadia Figueroa. A companion demo is available at [gpdf-demo](https://github.com/cr139139/gpdf-demo).

---

## What it does

Traditional obstacle representations (occupancy grids, point clouds) are discrete and difficult to differentiate through. GPDF treats a boundary point cloud as training data for a Gaussian Process and recovers a smooth, continuous distance field — complete with analytic gradients and Hessians — anywhere in the workspace. This makes it straightforward to plug into gradient-based controllers, CBF-based safety filters, and trajectory optimizers.

Key properties of the distance field:
- **Continuous and differentiable** everywhere, not just at grid cells
- **Gradient** of the distance is the inward/outward normal to the nearest boundary
- **Hessian** captures boundary curvature, useful for second-order planners
- Multiple obstacles are combined via a smooth log-sum-exp composition into a single unified field

---

## Features

- Fit a GPDF to any 2-D boundary described by a point cloud
- Query distance, gradient (surface normal), and Hessian at arbitrary query points
- Manage multiple obstacles through a single `GPDFEnv` environment object
- Automatic dense interpolation of sparse boundary polygons
- Contour and gradient-direction visualization via Matplotlib

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-org/GPDF_ADAS.git
cd GPDF_ADAS

# Install dependencies (Python 3.10+)
pip install jax jaxlib numpy matplotlib pillow
```

> The library uses JAX for JIT-compiled kernel computations and runs on CPU by default.

---

## Usage

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from gpdf.environment import GPDFEnv

# Define obstacle boundaries as ordered vertex lists
obstacle_list = [
    [(0, 0), (0, 1), (1, 1), (1, 0)],
    [(2, 2), (2, 3), (3, 3), (3, 2)],
]

# Build the environment — boundaries are auto-interpolated at 0.1 m resolution
env = GPDFEnv()
env.add_gpdfs_after_interp([0, 1], [np.asarray(obs) for obs in obstacle_list], interp_res=0.1)
print(f"Obstacles loaded: {env.num_gpdf}")

# Query distance and surface normal at a single point
query = np.array([[0.5, 0.5]])
dis, grad = env.h_grad_vector(query)
print(f"Distance: {dis}, Normal: {grad}")

# Visualize the unified distance field
fig, ax = plt.subplots()
env.plot_env(ax, x_range=(-1, 4), y_range=(-1, 4), show_grad=True)
for obs in obstacle_list:
    ax.add_patch(patches.Polygon(obs, closed=True, fill=True, color="gray", alpha=0.5))
ax.axis("equal")
plt.show()
```

Run the included demo directly:

```bash
cd src
python test_gpdf.py
```

---

## Project structure

```
GPDF_ADAS/
├── README.md
└── src/
    ├── test_gpdf.py          # Minimal end-to-end demo
    └── gpdf/
        ├── kernels.py        # Matérn-½ kernel, GP training, distance/gradient/Hessian inference
        ├── model.py          # GaussianProcessDistanceField — single-obstacle wrapper class
        └── environment.py    # GPDFEnv — multi-obstacle manager with smooth field composition
```

---

## Tech stack

| Component | Purpose |
|-----------|---------|
| [JAX](https://github.com/google/jax) | JIT-compiled kernel math and automatic differentiation |
| NumPy | Array manipulation and post-processing |
| Matplotlib | Distance field contour plots and gradient visualization |
| Pillow | Optional: load grayscale images as boundary targets |

The distance field uses a **Matérn-½ kernel** (equivalent to an Ornstein–Uhlenbeck covariance), which yields a distance-like decay that behaves well near obstacle surfaces. The length-scale `L = 0.2` controls how quickly correlation drops off with distance.

---

## License

MIT

