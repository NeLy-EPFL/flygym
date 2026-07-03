"""Canonical axis-name bindings for jaxtyping shape hints.

Referencing these (instead of bare string identifiers) lets pyflakes/ruff
resolve the forward references inside shape strings like
``Float[np.ndarray, "n_bodies"]`` instead of flagging them as undefined
names (F821) — no lint ignores needed.
"""

from typing import TypeVar

n_worlds = TypeVar("n_worlds")
n_jointdofs = TypeVar("n_jointdofs")
n_actuators = TypeVar("n_actuators")
n_tendon_actuators = TypeVar("n_tendon_actuators")
n_bodies = TypeVar("n_bodies")
n_sites = TypeVar("n_sites")
n_bodysegments = TypeVar("n_bodysegments")
n_cameras = TypeVar("n_cameras")
n_ommatidia = TypeVar("n_ommatidia")
