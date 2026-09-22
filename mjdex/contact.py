"""Contact-query helpers for MuJoCo environments.

These utilities follow the DualDex contact-checking pattern: cache sets of geom
IDs, then scan ``data.contact`` for pairwise intersections.
"""

from __future__ import annotations

from collections.abc import Iterable

import mujoco


def as_geom_id_set(geom_ids: Iterable[int]) -> set[int]:
    """Convert an iterable of geom IDs to a plain ``set[int]``."""

    return {int(gid) for gid in geom_ids}


def check_contact_geom_ids(
    data: mujoco.MjData,
    geoms_a: Iterable[int],
    geoms_b: Iterable[int] | None = None,
) -> bool:
    """Return True if geoms in ``geoms_a`` contact ``geoms_b``.

    If ``geoms_b`` is ``None``, this returns True when any geom in ``geoms_a`` is
    in contact with any other geom.
    """

    set_a = as_geom_id_set(geoms_a)
    set_b = None if geoms_b is None else as_geom_id_set(geoms_b)

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        geom1_in_a = geom1 in set_a
        geom2_in_a = geom2 in set_a

        if set_b is None:
            if geom1_in_a or geom2_in_a:
                return True
            continue

        geom1_in_b = geom1 in set_b
        geom2_in_b = geom2 in set_b
        if (geom1_in_a and geom2_in_b) or (geom2_in_a and geom1_in_b):
            return True

    return False


def collect_geoms_under_body_prefix(
    model: mujoco.MjModel,
    body_prefix: str,
    collision_only: bool = False,
) -> set[int]:
    """Collect geom IDs that belong to bodies whose names start with prefix."""

    body_ids = set()
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and body_name.startswith(body_prefix):
            body_ids.add(body_id)

    geom_ids = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) not in body_ids:
            continue
        if collision_only and not geom_can_collide(model, geom_id):
            continue
        geom_ids.append(geom_id)

    return set(geom_ids)


def collect_geoms_by_name_prefix(
    model: mujoco.MjModel,
    geom_prefix: str,
    collision_only: bool = False,
) -> set[int]:
    """Collect geom IDs whose geom names start with prefix."""

    geom_ids = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not geom_name or not geom_name.startswith(geom_prefix):
            continue
        if collision_only and not geom_can_collide(model, geom_id):
            continue
        geom_ids.append(geom_id)

    return set(geom_ids)


def geom_can_collide(model: mujoco.MjModel, geom_id: int) -> bool:
    """Return whether a geom participates in contact filtering."""

    return bool(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])


def contact_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geoms_a: Iterable[int] | None = None,
    geoms_b: Iterable[int] | None = None,
) -> list[tuple[str, str, float]]:
    """Return contact pairs as ``(geom1_name, geom2_name, distance)`` tuples."""

    set_a = None if geoms_a is None else as_geom_id_set(geoms_a)
    set_b = None if geoms_b is None else as_geom_id_set(geoms_b)
    pairs = []

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)

        if set_a is not None:
            geom1_in_a = geom1 in set_a
            geom2_in_a = geom2 in set_a
            if set_b is None:
                if not (geom1_in_a or geom2_in_a):
                    continue
            else:
                geom1_in_b = geom1 in set_b
                geom2_in_b = geom2 in set_b
                if not ((geom1_in_a and geom2_in_b) or (geom2_in_a and geom1_in_b)):
                    continue

        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or str(geom1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or str(geom2)
        pairs.append((name1, name2, float(contact.dist)))

    return pairs
