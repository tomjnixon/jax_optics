import jax.numpy as jnp
import numpy.testing as npt
from .transform import Transform

allclose_args = dict(
    rtol=1e-6,
    atol=1e-6,
)


def check_pos(t, p1, p2):
    p1, p2 = jnp.asarray(p1, jnp.float32), jnp.asarray(p2, jnp.float32)
    npt.assert_allclose(t.transform_forward(p1), p2, **allclose_args)
    npt.assert_allclose(t.transform_reverse(p2), p1, **allclose_args)


def check_vec(t, p1, p2):
    p1, p2 = jnp.asarray(p1, jnp.float32), jnp.asarray(p2, jnp.float32)
    npt.assert_allclose(t.transform_vec_forward(p1), p2, **allclose_args)
    npt.assert_allclose(t.transform_vec_reverse(p2), p1, **allclose_args)


def check_pos_and_vec(t, p1, p2):
    check_pos(t, p1, p2)
    check_vec(t, p1, p2)


def test_Transformed_translate():
    t = Transform().translate(1.0, 2.0, 3.0)

    check_pos(t, [0, 0, 0], [1, 2, 3])
    check_pos(t, [3, 4, 5], [4, 6, 8])
    check_vec(t, [1, 0, 0], [1, 0, 0])
    check_vec(t, [0, 1, 0], [0, 1, 0])
    check_vec(t, [0, 0, 1], [0, 0, 1])


def test_Transformed_rotate_z():
    t = Transform().rotate_z(jnp.pi / 4.0)

    root_half = jnp.sqrt(0.5)
    check_pos_and_vec(t, [0, 0, 0], [0, 0, 0])
    check_pos_and_vec(t, [1, 0, 0], [root_half, root_half, 0])
    check_pos_and_vec(t, [0, 1, 0], [-root_half, root_half, 0])
    check_pos_and_vec(t, [root_half, root_half, 0], [0, 1, 0])


def test_Transformed_rotate_translate():
    t = Transform().rotate_z(jnp.pi / 4.0).translate(1.0, 2.0, 3.0)

    root_half = jnp.sqrt(0.5)
    check_vec(t, [0, 0, 0], [0, 0, 0])
    check_vec(t, [1, 0, 0], [root_half, root_half, 0])
    check_vec(t, [0, 1, 0], [-root_half, root_half, 0])
    check_vec(t, [0, 0, 1], [0, 0, 1])
    check_vec(t, [root_half, root_half, 0], [0, 1, 0])

    check_pos(t, [0, 0, 0], [1, 2, 3])

    check_pos(t, [1, 0, 0], [1 + root_half, 2 + root_half, 3])
    check_pos(t, [0, 0, 1], [1, 2, 4])
