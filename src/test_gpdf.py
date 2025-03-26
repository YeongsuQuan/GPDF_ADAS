import numpy as np
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as patches # type: ignore

from basic_boundary_function.env import GPDFEnv


obstacle_list = [
    [(0, 0), (0, 1), (1, 1), (1, 0)],
    [(2, 2), (2, 3), (3, 3), (3, 2)],
]

env = GPDFEnv()
env.add_gpdfs_after_interp([0, 1], [np.asarray(x) for x in obstacle_list], 0.1)
print(f'Number of GPDFs loaded: {env.num_gpdf}')

fig, ax = plt.subplots()
env.plot_env(ax, x_range=(-1, 4), y_range=(-1, 4), show_grad=True)
for obs in obstacle_list:
    ax.add_patch(patches.Polygon(obs, closed=True, fill=True, color='gray', alpha=0.5))
ax.axis('equal')
plt.show()