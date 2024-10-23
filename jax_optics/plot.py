import jax
import matplotlib.pyplot as plt


def plot_rays(positions: jax.Array, ax=None, **kwargs):
    """plot rays from a matrix of positions, which should be shape (n, k, 3)
    for n rays with k positions each (for k-1 surfaces)"""
    if ax is None:
        ax = plt.gca()

    ax.plot(positions[:, :, 0].T, positions[:, :, 1].T, **kwargs)


def plot_surface(surface, ax=None, n=50, **kwargs):
    """plot the shape of a surface"""
    if ax is None:
        ax = plt.gca()

    points = surface.get_xy_points(n)
    ax.plot(points[:, 0], points[:, 1], **kwargs)
