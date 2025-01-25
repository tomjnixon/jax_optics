import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
@partial(
    jnp.vectorize,
    excluded={2, "surfaces"},
    signature="(3),(3)->(n,3),(m,3)",
)
def trace(pos, dir, surfaces):
    """trace paths of a ray through a list of surfaces

    Parameters
    ----------
    pos
        3D ray start position
    dir
        3D ray start direction
    surfaces
        n tuples containing a Surface and Interaction for that surface

    Returns
    -------
    positions
        array of ray start / corner / end positions
    directions
        array of ray directions

    the ray starting at positions[i] has direction directions[i]

    if any interactions are Stop, positions will have one more element than
    directions, and the last element of positions will be on the Stop surface.
    otherwise, positions and directions will have n+1 elements
    """
    positions = [pos]
    directions = [dir]

    for surface, interaction in surfaces:
        pos, normal = surface.intersect(pos, dir)

        dir = interaction.interact(pos, dir, normal)

        positions.append(pos)
        if dir is None:
            break
        directions.append(dir)

    return jnp.array(positions), jnp.array(directions)
