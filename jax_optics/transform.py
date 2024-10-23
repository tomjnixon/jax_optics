import jax
import jax.numpy as jnp
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Self


def _uniform_transform(M, pos):
    res = M @ jnp.append(pos, 1.0)
    return res[:-1] / res[-1]


def wrap_transform_2d(transform_fun):
    def wrapped(pos):
        return transform_fun(jnp.append(pos, 1.0))[:-1]

    return wrapped


@dataclass
class Transform:
    transformation_matrix: jax.Array = field(
        kw_only=True, default_factory=lambda: jnp.eye(4)
    )

    @jax.jit
    def transform_forward(self, pos: jax.Array) -> jax.Array:
        return _uniform_transform(self.transformation_matrix, pos)

    @jax.jit
    def transform_reverse(self, pos: jax.Array) -> jax.Array:
        # TODO: would it be worth it to store an inverse transformation matrix
        # (updated with the inverse of each pushed transformation, which is
        # normally obvious) to avoid this invert?
        return _uniform_transform(jnp.linalg.inv(self.transformation_matrix), pos)

    @jax.jit
    def transform_vec_forward(self, vec: jax.Array) -> jax.Array:
        return self.transformation_matrix[:3, :3] @ vec

    @jax.jit
    def transform_vec_reverse(self, vec: jax.Array) -> jax.Array:
        return jnp.linalg.inv(self.transformation_matrix[:3, :3]) @ vec

    @jax.jit
    def push(self, M: jax.Array) -> Self:
        return replace(self, transformation_matrix=M @ self.transformation_matrix)

    @jax.jit
    def translate(self, x=0.0, y=0.0, z=0.0) -> Self:
        return self.push(
            jnp.array(
                [
                    [1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, y],
                    [0.0, 0.0, 1.0, z],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

    @partial(jax.jit, static_argnames=["axis"])
    def rotate_ax(self, theta: float, axis: int) -> Self:
        # there is definitely a shorter way to express this, but this way should
        # result in a tidy jax expression
        m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        ax_idx = [0, 1, 2]
        ax_idx.remove(axis)

        s, c = jnp.sin(theta), jnp.cos(theta)

        m[ax_idx[0]][ax_idx[0]] = c
        m[ax_idx[0]][ax_idx[1]] = -s
        m[ax_idx[1]][ax_idx[0]] = s
        m[ax_idx[1]][ax_idx[1]] = c

        return self.push(jnp.array(m))

    def rotate_x(self, theta: float) -> Self:
        return self.rotate_ax(theta, 0)

    def rotate_y(self, theta: float) -> Self:
        return self.rotate_ax(theta, 1)

    def rotate_z(self, theta: float) -> Self:
        return self.rotate_ax(theta, 2)


# for use in subclasses
Transform_field_names = ["transformation_matrix"]

jax.tree_util.register_dataclass(
    Transform, data_fields=Transform_field_names, meta_fields=[]
)
