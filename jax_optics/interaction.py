import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional


class Interaction:
    """specifies the way that a surface changes the properties of a ray"""

    def interact(
        self, pos: jax.Array, dir: jax.Array, normal: jax.Array
    ) -> Optional[jax.Array]:
        """specify how a ray changes direction when it hits a surface

        Parameters
        ----------
        pos
            position on the surface
        dir
            normalised direction vector of the ray
        normal
            normal of the surface at pos

        Returns
        -------
        Optional[jax.Array]
            new direction of the ray, or None if the ray stops
        """
        raise NotImplementedError()


@dataclass
class Stop(Interaction):
    def interact(self, pos, dir, normal):
        return None


jax.tree_util.register_dataclass(Stop, data_fields=[], meta_fields=[])


@dataclass
class Refract(Interaction):
    """refractive behaviour of a surface

    the normal of the surface should point from from a region with refractive
    index n_inside to a region with refractive index n_outside

    """

    n_outside: float
    n_inside: float

    @jax.jit
    def interact(self, pos, dir, normal):
        # this is a but messy; detect which direction the ray is going in, and
        # normalise the refractive indices and normal
        reverse = jnp.dot(dir, normal) > 0
        n_in = jnp.where(reverse, self.n_inside, self.n_outside)
        n_out = jnp.where(reverse, self.n_outside, self.n_inside)
        normal = jnp.where(reverse, normal, -normal)

        # from https://physics.stackexchange.com/a/436252
        mu = n_in / n_out
        ni = normal @ dir
        return jnp.sqrt(1 - mu * mu * (1 - ni * ni)) * normal + mu * (dir - ni * normal)


jax.tree_util.register_dataclass(
    Refract, data_fields=["n_outside", "n_inside"], meta_fields=[]
)
