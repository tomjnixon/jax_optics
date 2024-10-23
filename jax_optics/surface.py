import jax
import jax.numpy as jnp
from .transform import Transform, Transform_field_names, wrap_transform_2d
from dataclasses import dataclass


@jax.jit
def convolve_unrolled_full(a, b):
    """equivalent to jnp.convolve(a, b, mode="full"), but unrolled to work
    better with small arrays (which polynomial coefficients normally are)"""

    def at_i(i):
        return sum(a[ai] * b[i - ai] for ai in range(len(a)) if 0 <= i - ai < len(b))

    return jnp.array([at_i(i) for i in range(len(a) + len(b) - 1)])


def polymul(a, b):
    """same as jnp.polymul, but uses convolve_unrolled_full"""
    return convolve_unrolled_full(a, b)


def poly_substitute(p_outer: jax.Array, p_inner: jax.Array) -> jax.Array:
    """substitute one polynomial into another

    polynomials are defined by their coefficients (in the order expected by
    polyval etc.); this returns coefficients equivalent to p_outer(p_inner(x)).
    """
    y = jnp.array([])
    for pv in p_outer:
        if len(y) == 0:
            y = jnp.array([pv])
        else:
            y = jnp.polyadd(polymul(y, p_inner), jnp.array([pv]))
    return y


def first_real_pos_root(poly: jax.Array, select=None) -> jax.Array:
    """find the first real non-negative (with some tolerance_ root of a polynomial"""
    roots = jnp.roots(poly, strip_zeros=False)

    full_select = jnp.isreal(roots) & (jnp.real(roots) > -1e-6)
    if select is not None:
        full_select = full_select & jax.vmap(select)(roots)

    return jnp.min(jnp.real(roots), where=full_select, initial=jnp.inf)


def norm_vector(v: jax.Array) -> jax.Array:
    return v / jnp.linalg.norm(v)


class Surface(Transform):
    """defines the shape (but not properties) of some surface in an optical model"""

    def intersect(self, pos: jax.Array, dir: jax.Array) -> tuple[jax.Array, jax.Array]:
        """find the intersection of a ray with the surface

        Parameters
        ----------
        pos
            start position of the ray
        dir
            normalised direction vector of the ray

        Returns
        -------
        pos : jax.Array
            position of intersection (always nan or inf if no intersection)
        normal : jax.Array
            normal of surface at intersection (possibly nan or inf if no intersection)
        """
        # developer notes on this design

        # yes, this choice of return value is a bit messy, as the point is
        # identified by a distance along the ray, and the normal is not really
        # a property of an intersection

        # this is done because different surface shapes are best represented by
        # different surface models, and the information that we would want to
        # pass between an idealised `intersect` and `normal` function may be
        # different for different models

        # for example, for a surface defined as a function in (y, z), the
        # normal can be efficiently computed from the (y, z) position. on the
        # other hand, for a parametric surface (where x=fx(t, u), y=fy(t, u)
        # and z=fz(t, u)), we may have a value for t and u at the intersection,
        # and if this is thrown away, calculating the normal becomes much more
        # difficult than necessary (e.g. it might need an iterative solution
        # where otherwise it could be a simple closed-form expression)

        # this doesn't stop us from producing a better API for each underlying
        # model, with this defined in a base class, if desired

        # finally, the position should be differentiable wrt. the direction, so
        # it would be possible to derive a normal from the position alone, but
        # it's often trivial to calculate the normal directly, so it's worth
        # doing that

        int_pos, normal = self._intersect_raw(
            self.transform_reverse(pos),
            self.transform_vec_reverse(dir),
        )

        return self.transform_forward(int_pos), self.transform_vec_forward(normal)

    def _intersect_raw(
        self, pos: jax.Array, dir: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """same as intersect but on local coordinates, for implementation in subclasses"""
        raise NotImplementedError()

    def get_xy_points(self, n: int) -> jax.Array:
        """get points in the xy plane for plotting"""
        points = self._get_xy_points_raw(n)

        return jax.vmap(wrap_transform_2d(self.transform_forward))(points)

    def _get_xy_points_raw(self, n):
        raise NotImplementedError()


class Shape:
    """a 2D shape (generally in the yz plane) used to clip a surface"""

    def hit(self, pos: jax.Array) -> jax.Array:
        """is a 2D position (YZ) within this shape?"""
        raise NotImplementedError()

    def y_extent(self) -> tuple[float, float]:
        """get the lower and upper limit in y for plotting"""
        raise NotImplementedError()


@dataclass
class Circle(Shape):
    r: float

    def hit(self, pos: jax.Array) -> jax.Array:
        return (pos @ pos) <= self.r * self.r

    def y_extent(self):
        return (-self.r, self.r)


jax.tree_util.register_dataclass(Circle, data_fields=["r"], meta_fields=[])


@dataclass
class Square(Shape):
    size: float

    def hit(self, pos: jax.Array) -> jax.Array:
        return jnp.max(jnp.abs(pos)) <= self.size / 2.0

    def y_extent(self):
        return (-self.size / 2, self.size / 2)


jax.tree_util.register_dataclass(Square, data_fields=["size"], meta_fields=[])


@dataclass
class Plane(Surface):
    """plane defined by x=0"""

    shape: Shape

    @jax.jit
    def _intersect_raw(self, pos, dir):
        dist = -pos[0] / dir[0]

        new_pos = pos + dist * dir

        right_dir = dist > -1e-6
        hit_shape = self.shape.hit(new_pos[1:])

        new_pos = jnp.where(right_dir & hit_shape, new_pos, jnp.repeat(jnp.inf, 3))
        return new_pos, jnp.array([-1.0, 0.0, 0.0])

    def _get_xy_points_raw(self, n):
        y_min, y_max = self.shape.y_extent()
        return jnp.array(
            [
                [0.0, y_min],
                [0.0, y_max],
            ]
        )


jax.tree_util.register_dataclass(
    Plane,
    data_fields=["shape"] + Transform_field_names,
    meta_fields=[],
)


@dataclass
class Sphere(Surface):
    """spherical surface: half of a sphere (closest to [0, 0, 0]) with radius
    |r| and centre [r, 0, 0]
    """

    r: float
    shape: Shape

    @jax.jit
    def _intersect_raw(self, pos, dir):
        centre = jnp.array([self.r, 0.0, 0.0])
        r_sign = jnp.sign(self.r)
        r = jnp.abs(self.r)
        # adapted from
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        rel_pos = pos - centre
        a = jnp.dot(dir, dir)
        b = 2.0 * jnp.dot(dir, rel_pos)
        c = jnp.dot(rel_pos, rel_pos) - r * r

        def select(root):
            int_pos = pos + root * dir
            correct_half = r_sign * int_pos[0] < r
            in_shape = self.shape.hit(int_pos[1:])
            return correct_half & in_shape

        dist = first_real_pos_root(jnp.array([a, b, c]), select=select)

        new_pos = pos + dist * dir
        return new_pos, r_sign * norm_vector(new_pos - centre)

    def _get_xy_points_raw(self, n):
        centre = jnp.array([self.r, 0.0])
        r_abs = jnp.abs(self.r)

        y_min, y_max = self.shape.y_extent()

        sin_max_angle = y_max / r_abs
        sin_min_angle = y_min / r_abs

        max_angle = jnp.arcsin(jnp.minimum(sin_max_angle, 1.0))
        min_angle = jnp.arcsin(jnp.maximum(sin_min_angle, -1.0))

        angles = jnp.linspace(min_angle, max_angle, n)

        points_raw = jnp.stack((-jnp.cos(angles), jnp.sin(angles)), axis=1)

        return centre + self.r * points_raw


jax.tree_util.register_dataclass(
    Sphere,
    data_fields=["r", "shape"] + Transform_field_names,
    meta_fields=[],
)


@dataclass
class SpherePolyMod(Surface):
    """spherical surface offset along x by a polynomial in the squared distance
    from the x axis

    This includes points which are on the offset sphere:

        (x - r - p(y**2 + z**2)) ** 2 + y**2 + z**2 = r**2

    and in the half closest to the origin:

        |(x - p(y**2 + x**2))| < |r|

    p(r2) is defined by coeffs, which can be thought of as a polynomial in the
    radius squared, or the odd coefficients of a polynomial in r
    """

    r: float
    shape: Shape
    coeffs: jax.Array

    @jax.jit
    def _intersect_raw(self, pos, dir):
        r_sign = jnp.sign(self.r)
        r = jnp.abs(self.r)

        # x y and z pos expressed as polynomials in t
        x = jnp.array([dir[0], pos[0] - self.r])
        y = jnp.array([dir[1], pos[1]])
        z = jnp.array([dir[2], pos[2]])

        y2, z2 = polymul(y, y), polymul(z, z)

        ryz2 = jnp.polyadd(y2, z2)

        # offset in x from sphere expressed as polynomial in t
        x_poly = poly_substitute(self.coeffs, ryz2)

        # x position on sphere after removing offset
        x_sphere = jnp.polysub(x, x_poly)
        x_sphere2 = polymul(x_sphere, x_sphere)

        r2 = jnp.polyadd(jnp.polyadd(x_sphere2, y2), z2)

        to_solve = jnp.polysub(r2, jnp.array([r * r]))

        def select(root):
            int_pos = pos + root * dir

            ryz2 = int_pos[1] * int_pos[1] + int_pos[2] * int_pos[2]
            x_sphere = int_pos[0] - jnp.polyval(self.coeffs, ryz2)

            correct_half = r_sign * x_sphere < r
            in_shape = self.shape.hit(int_pos[1:])
            return correct_half & in_shape

        dist = first_real_pos_root(to_solve, select=select)

        new_pos = pos + dist * dir

        # calculate normal by differentiating the function from yz to x
        # coordinates

        def yz_to_x(yz):
            r2 = yz[0] * yz[0] + yz[1] * yz[1]

            x_sphere = r_sign * (r - jnp.sqrt(r * r - r2))
            x_poly = jnp.polyval(self.coeffs, r2)

            return x_poly + x_sphere

        jac_yz = jax.jacobian(yz_to_x)(new_pos[1:])
        norm = norm_vector(jnp.concatenate((jnp.array([-1]), jac_yz)))

        return new_pos, norm

    def _get_xy_points_raw(self, n):
        centre = jnp.array([self.r, 0.0])
        r_abs = jnp.abs(self.r)

        y_min, y_max = self.shape.y_extent()

        sin_max_angle = y_max / r_abs
        sin_min_angle = y_min / r_abs

        max_angle = jnp.arcsin(jnp.minimum(sin_max_angle, 1.0))
        min_angle = jnp.arcsin(jnp.maximum(sin_min_angle, -1.0))

        angles = jnp.linspace(min_angle, max_angle, n)

        points_sphere = self.r * jnp.stack((-jnp.cos(angles), jnp.sin(angles)), axis=1)

        r2 = points_sphere[:, 1] * points_sphere[:, 1]
        offset_x = jnp.polyval(self.coeffs, r2)
        offset = jnp.stack((offset_x, jnp.zeros(n)), axis=1)

        return centre + points_sphere + offset


jax.tree_util.register_dataclass(
    SpherePolyMod,
    data_fields=["r", "shape", "coeffs"] + Transform_field_names,
    meta_fields=[],
)


@dataclass
class PolySurface(Surface):
    """surface defined by a polynomial in the squared distance from the x axis,
    such that:

    x = polyval(coeffs, y ** 2 + z ** 2)
    """

    coeffs: jax.Array
    shape: Shape

    @jax.jit
    def _intersect_raw(self, pos, dir):
        x = jnp.array([dir[0], pos[0]])
        y = jnp.array([dir[1], pos[1]])
        z = jnp.array([dir[2], pos[2]])

        r2 = jnp.polyadd(polymul(y, y), polymul(z, z))

        # coeffs contains even coefficients, so if we substitute r^2 into it,
        # it works out the same as if we substituted r into an coeffs after
        # expansion
        x_t = poly_substitute(self.coeffs, r2)

        def select(root):
            int_pos = pos + root * dir
            return self.shape.hit(int_pos[1:])

        dist = first_real_pos_root(jnp.polysub(x_t, x), select=select)
        int_pos = pos + dir * dist

        def yz_to_x(yz):
            r2 = yz[0] * yz[0] + yz[1] * yz[1]
            return jnp.polyval(self.coeffs, r2)

        jac_yz = jax.jacobian(yz_to_x)(int_pos[1:])

        norm = norm_vector(jnp.concatenate((jnp.array([-1.0]), jac_yz)))

        return int_pos, norm

    def _get_xy_points_raw(self, n):
        y_min, y_max = self.shape.y_extent()

        ys = jnp.linspace(y_min, y_max, n)

        xs = jnp.polyval(self.coeffs, ys * ys)

        return jnp.stack((xs, ys), axis=1)


jax.tree_util.register_dataclass(
    PolySurface,
    data_fields=["coeffs", "shape"] + Transform_field_names,
    meta_fields=[],
)
