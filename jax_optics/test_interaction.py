import jax.numpy as jnp
import numpy.testing as npt
import pytest
from .interaction import Refract


def angle(theta, axis=-1):
    """direction vector anticlockwise from x axis"""
    return jnp.stack(
        (jnp.cos(jnp.radians(theta)), jnp.sin(jnp.radians(theta))), axis=-1
    )


@pytest.mark.parametrize("flip", [False, True])
def test_Refract(flip):
    r = Refract(1, 2)

    norm_angle = 10.0
    in_angle = 20.0

    if flip:
        n_in, n_out = r.n_inside, r.n_outside
        norm = angle(norm_angle)
    else:
        n_in, n_out = r.n_outside, r.n_inside
        norm = angle(180.0 + norm_angle)

    out_angle = jnp.degrees(jnp.arcsin(jnp.sin(jnp.radians(in_angle)) * (n_in / n_out)))

    dir_in = angle(norm_angle + in_angle)
    dir_out_expected = angle(norm_angle + out_angle)

    dir_out = r.interact(jnp.array([0.0, 0.0]), dir_in, norm)

    npt.assert_allclose(dir_out, dir_out_expected)
