import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from .surface import (
    Circle,
    Plane,
    PolySurface,
    Sphere,
    SpherePolyMod,
    Square,
    convolve_unrolled_full,
    norm_vector,
    poly_substitute,
)
from .transform import Transform
from numpy.polynomial import Polynomial as np_Polynomial


def test_convolve_unrolled_full():
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])

    c = convolve_unrolled_full(a, b)
    c_ex = jnp.convolve(a, b, mode="full")
    import numpy.testing as npt

    npt.assert_allclose(c, c_ex)


def test_poly_substitute():
    p = [1, 2, 3]
    x = [4, 5]

    expected = np_Polynomial(p[::-1])(np_Polynomial(x[::-1])).coef[::-1]

    npt.assert_allclose(poly_substitute(jnp.array(p), jnp.array(x)), expected)


def test_Circle():
    c = Circle(1.0)

    assert c.hit(jnp.array([0.5, 0.5]))
    assert not c.hit(jnp.array([1.0, 1.0]))
    assert c.hit(jnp.array([-0.5, 0.5]))
    assert not c.hit(jnp.array([-1.0, 1.0]))


def test_Square():
    c = Square(2.0)

    assert c.hit(jnp.array([0.9, 0.9]))
    assert c.hit(jnp.array([-0.9, 0.9]))

    assert not c.hit(jnp.array([0.9, 1.1]))
    assert not c.hit(jnp.array([0.9, -1.1]))


@jax.jit
def grad_norm(s, pos, dir):
    """get the normal of a surface at an intersection with a ray by
    differentiating its intersect function
    """
    pos_jac, _norm = jax.jacobian(s.intersect, has_aux=True)(pos, dir)

    return norm_vector(jnp.cross(pos_jac[:, 2], pos_jac[:, 1]))


@jax.jit
def surf_dist(s, pos):
    pos = s.transform_reverse(pos)

    if isinstance(s, Sphere):
        centre = jnp.array([s.r, 0.0, 0.0])
        r = jnp.abs(s.r)

        return jnp.linalg.norm(pos - centre) - r

    elif isinstance(s, SpherePolyMod):
        centre = jnp.array([s.r, 0.0, 0.0])
        r = jnp.abs(s.r)

        ryz = jnp.linalg.norm(pos[1:])
        n = len(s.coeffs) * 2 - 1
        coeffs = jnp.zeros(n).at[::2].set(s.coeffs)
        x_poly = jnp.polyval(coeffs, ryz)
        poly_offset = jnp.array([x_poly, 0.0, 0.0])

        return jnp.linalg.norm(pos - (centre + poly_offset)) - r

    elif isinstance(s, Plane):
        return -pos[0]

    elif isinstance(s, PolySurface):
        n = len(s.coeffs) * 2 - 1
        coeffs = jnp.zeros(n).at[::2].set(s.coeffs)

        r = jnp.linalg.norm(pos[1:])
        return jnp.polyval(coeffs, r) - pos[0]

    else:
        assert False


def check_ray_3d(s, pos, dir, expected_hit):
    pos, dir = jnp.asarray(pos, jnp.float32), jnp.asarray(dir, jnp.float32)

    i_pos, norm = s.intersect(pos, dir)

    hit = bool(jnp.all(jnp.isfinite(i_pos)))
    assert hit == expected_hit

    if hit:
        assert (
            jnp.linalg.norm(jnp.cross(i_pos - pos, dir)) < 1e-6
        ), "intersection is not on ray"

        assert jnp.dot(i_pos - pos, dir) > -1e-6, "intersection is in wrong direction"

        dist = surf_dist(s, i_pos)
        assert jnp.abs(dist) < 1e-6

        expected_norm = grad_norm(s, pos, dir)
        # normal is allowed to point in either direction, while grad_norm
        # always points towards pos
        if jnp.dot(norm, expected_norm) < 0:
            expected_norm = -expected_norm

        npt.assert_allclose(norm, expected_norm, rtol=1e-6, atol=1e-6)

    return i_pos, norm


def check_ray(s, pos, dir, expected_hit):
    """check properties of a ray-surface intersection which is rotationally
    symmetrical around the x axis.

    pos and dir are 2d point/vector in the xy plane
    """
    pos, dir = jnp.asarray(pos, jnp.float32), jnp.asarray(dir, jnp.float32)

    pos, dir = jnp.append(pos, 0.0), jnp.append(dir, 0.0)

    results_direct = check_ray_3d(s, pos, dir, expected_hit)

    for angle in jnp.pi / 4.0, jnp.pi / 2.0:
        t = Transform().rotate_x(angle)
        pos_t = t.transform_forward(pos)
        dir_t = t.transform_vec_forward(dir)

        check_ray_3d(s, pos_t, dir_t, expected_hit)

    return results_direct


def check_points(s, n=21):
    """check that the points returned by get_xy_points are on the surface"""
    points = s.get_xy_points(n)
    points_3d = jnp.append(points, jnp.zeros((points.shape[0], 1)), axis=1)
    distances = jax.vmap(surf_dist, (None, 0))(s, points_3d)
    npt.assert_allclose(distances, 0.0, atol=1e-6)


def test_Plane():
    p = Plane(Circle(3.0)).translate(3.0)

    check_ray(p, [0.0, 2.0], [1.0, 0.0], True)
    check_ray(p, [0.0, 4.0], [1.0, 0.0], False)

    check_ray(p, [5.0, 2.0], [-1.0, 0.0], True)

    root_half = np.sqrt(0.5)
    check_ray(p, [2.0, 0.0], [root_half, root_half], True)

    check_ray(p, [0.0, 0.0], [-1.0, 0.0], False)

    check_points(p)


def test_Sphere():
    # pointing left
    s = Sphere(1.0, Circle(2.0)).translate(1.0)

    check_ray(s, [0.0, 0.0], [1.0, 0.0], True)
    check_ray(s, [0.0, 0.5], [1.0, 0.0], True)

    root_half = np.sqrt(0.5)

    check_ray(s, [0.0, 0.0 + root_half], [1.0, 0.0], True)

    check_ray(s, [1.0, 1.0], [root_half, -root_half], True)

    # miss
    check_ray(s, [0.0, 2.0], [1.0, 0.0], False)
    check_ray(s, [2.0, 0.0], [1.0, 0.0], False)

    check_points(s)

    # pointing right
    s = Sphere(-1.0, Circle(2.0)).translate(1.0)

    check_ray(s, [0.0, 0.0], [1.0, 0.0], True)

    check_ray(s, [0.0, 0.0], [root_half, root_half], True)

    check_points(s)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_SpherePolyMod_no_mod(sign):
    s = SpherePolyMod(sign * 1.0, Circle(2.0), jnp.array([0.0])).translate(1.0)

    check_ray(s, [0.0, 0.0], [1.0, 0.0], True)
    check_ray(s, [0.0, 0.5], [1.0, 0.0], True)

    check_ray(s, [0.0, 0.0], [1.0, 0.1], True)
    check_ray(s, [0.0, 0.5], [1.0, -0.1], True)

    check_points(s)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_SpherePolyMod_offset(sign):
    s = SpherePolyMod(sign * 1.0, Circle(2.0), jnp.array([2.0])).translate(1.0)

    check_ray(s, [0.0, 0.0], [1.0, 0.0], True)
    check_ray(s, [0.0, 0.5], [1.0, 0.0], True)

    check_ray(s, [0.0, 0.0], [1.0, 0.1], True)
    check_ray(s, [0.0, 0.5], [1.0, -0.1], True)

    check_points(s)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_SpherePolyMod_square(sign):
    s = SpherePolyMod(sign * 1.0, Circle(2.0), jnp.array([0.5, 0.0])).translate(1.0)

    check_ray(s, [0.0, 0.0], [1.0, 0.0], True)
    check_ray(s, [0.0, 0.5], [1.0, 0.0], True)

    check_ray(s, [0.0, 0.0], [1.0, 0.1], True)
    check_ray(s, [0.0, 0.5], [1.0, -0.1], True)

    check_points(s)


def test_PolySurface():
    # vertical plane
    s = PolySurface(jnp.array([2.0]), Circle(3.0))
    check_ray(s, [0.0, 2.0], [1.0, 0.0], True)
    check_points(s)

    # quadratic
    s = PolySurface(jnp.array([1.0, 2.0]), Circle(3.0))
    check_ray(s, [0.0, 0.0], [1.0, 0.0], True)
    check_ray(s, [0.0, 1.0], [1.0, 0.0], True)

    root_half = np.sqrt(0.5)
    # angled ray
    check_ray(s, [0.0, 3.0], [root_half, -root_half], True)

    check_points(s)
