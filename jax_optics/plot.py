import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_rays(positions: jax.Array, ax=None, **kwargs):
    """plot rays from a matrix of positions, which should be shape (..., k, 3)
    for an arbitrary structure of rays with k positions each (for k-1 surfaces)
    """
    if ax is None:
        ax = plt.gca()

    *_rest, k, dim = positions.shape
    assert dim == 3

    pos_flat = jnp.reshape(positions, (-1, k, dim))

    ax.plot(pos_flat[..., 0].T, pos_flat[..., 1].T, **kwargs)


def plot_surface(surface, ax=None, n=50, **kwargs):
    """plot the shape of a surface"""
    if ax is None:
        ax = plt.gca()

    points = surface.get_xy_points(n)
    ax.plot(points[:, 0], points[:, 1], **kwargs)
