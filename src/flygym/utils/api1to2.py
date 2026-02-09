import re

__all__ = ["BODY_NAMES_OLD2NEW", "BODY_NAMES_NEW2OLD"]


def _body_name_old2new(old_name: str) -> str:
    # Case 1: center-line segments
    center_regex = r"^(Thorax|Head|Rostrum|Haustellum|A1A2|A3|A4|A5|A6)$"
    if (match := re.match(center_regex, old_name)) is not None:
        seg = match.group(1)
        if seg == "A1A2":
            seg = "abdomen12"
        if seg in ("A3", "A4", "A5", "A6"):
            seg = f"abdomen{seg[-1]}"
        return f"c_{seg.lower()}"

    # Case 2: leg segments
    leg_regex = r"^([LR][FMH])(Coxa|Femur|Tibia|Tarsus[1-5])$"
    if (match := re.match(leg_regex, old_name)) is not None:
        leg, seg = match.groups()
        if seg == "Femur":
            seg = "trochanterfemur"
        return f"{leg.lower()}_{seg.lower()}"

    # Case 3: other sided segments
    other_sided_regex = r"^([LR])(Eye|Pedicel|Funiculus|Arista|Haltere|Wing)$"
    if (match := re.match(other_sided_regex, old_name)) is not None:
        side, seg = match.groups()
        return f"{side.lower()}_{seg.lower()}"

    raise ValueError(f"Unknown legacy body name: {old_name}")


# fmt: off
_OLD_CENTER_SEGS = ("Thorax", "Head", "Rostrum", "Haustellum", "A1A2", "A3", "A4", "A5", "A6")
_OLD_SIDED_SEGS = ("Eye", "Pedicel", "Funiculus", "Arista", "Haltere", "Wing")
_OLD_LEG_SEGS = ("Coxa", "Femur", "Tibia", *(f"Tarsus{i}" for i in range(1, 6)))
_OLD_BODY_NAMES = [
    *_OLD_CENTER_SEGS,
    *(f"{side}{seg}" for side in "LR" for seg in _OLD_SIDED_SEGS),
    *(f"{side}{pos}{seg}" for side in "LR" for pos in "FMH" for seg in _OLD_LEG_SEGS),
]
BODY_NAMES_OLD2NEW = {old: _body_name_old2new(old) for old in _OLD_BODY_NAMES}
BODY_NAMES_NEW2OLD = {new: old for old, new in BODY_NAMES_OLD2NEW.items()}
# fmt: on
