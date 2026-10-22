"""Generate `rigging_separate_trochanterfemur.yaml` from `rigging.yaml`.

This is a maintenance/dev script, not part of the runtime package. It derives the
rigging config of `NeuroMechFlyWithSeparateTrochanterFemur` from that of
`NeuroMechFly` by replacing each `{leg}_trochanterfemur` entry with a
`{leg}_trochanter` and a `{leg}_femur` entry:

- The trochanter takes over the pose of the trochanterfemur.
- The femur is placed relative to the trochanter at the trochanter-femur boundary,
  which is also the origin of the femur mesh.
- The tibia, now a child of the femur, is shifted so that its pose in the world is
  unchanged.
- The mass of the trochanterfemur is split between the trochanter and the femur.

The femur offsets and masses come from commit e8140d4 ("splitting the femur and
the trochanter geom for valid segmentation maps"), where they were defined for the
left legs. The right legs mirror them across the XZ plane.

The edits are text substitutions, so the rest of the file (order, formatting,
comments) is unchanged. Re-run this script whenever `rigging.yaml` changes.

Usage:
    python scripts/dev/make_rigging_separate_trochanterfemur.py
"""

import re

import yaml

from flygym import assets_dir

SRC_PATH = assets_dir / "model/neuromechfly/rigging.yaml"
DST_PATH = assets_dir / "model/neuromechfly/rigging_separate_trochanterfemur.yaml"

# Per left leg: femur origin in the trochanter frame (mm), trochanter mass, femur mass
SPLIT_BY_LEFT_LEG = {
    "lf": ([0.0, 0.042, -0.129], 7.35180e-07, 5.55480e-06),
    "lm": ([0.036, -0.028, -0.109], 8.13140e-07, 5.66690e-06),
    "lh": ([0.025, 0.026, -0.125], 6.03930e-07, 8.14610e-06),
}
IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]


def _format_vec(vec: list[float]) -> str:
    # Round away float noise from the subtraction; `+ 0.0` turns -0.0 into 0.0
    return "[" + ", ".join(repr(round(x, 6) + 0.0) for x in vec) + "]"


def make_rigging(text: str) -> str:
    """Return the separate-trochanterfemur rigging YAML given the fused one."""
    config = yaml.safe_load(text)
    for leg in ["lf", "lm", "lh", "rf", "rm", "rh"]:
        femur_pos, trochanter_mass, femur_mass = SPLIT_BY_LEFT_LEG[f"l{leg[1]}"]
        if leg[0] == "r":
            femur_pos = [femur_pos[0], -femur_pos[1], femur_pos[2]]

        # Shifting the tibia by `femur_pos` is only valid without rotations
        trochanterfemur = config[f"{leg}_trochanterfemur"]
        if trochanterfemur["quat"] != IDENTITY_QUAT:
            raise ValueError(f"{leg}_trochanterfemur must have an identity quat")
        if abs(trochanter_mass + femur_mass - trochanterfemur["mass"]) > 1e-10:
            raise ValueError(f"Split masses of {leg} do not add up to the total")

        old_block = re.search(
            rf"^{leg}_trochanterfemur:\n(  .*\n)+", text, flags=re.M
        ).group(0)
        new_block = (
            f"{leg}_trochanter:\n"
            f"  pos: {_format_vec(trochanterfemur['pos'])}\n"
            f"  quat: {IDENTITY_QUAT}\n"
            f"  mass: {trochanter_mass:.5e}\n"
            f"{leg}_femur:\n"
            f"  pos: {_format_vec(femur_pos)}\n"
            f"  quat: {IDENTITY_QUAT}\n"
            f"  mass: {femur_mass:.5e}\n"
        )
        text = text.replace(old_block, new_block)

        tibia_pos = config[f"{leg}_tibia"]["pos"]
        new_tibia_pos = [a - b for a, b in zip(tibia_pos, femur_pos)]
        text, n_subs = re.subn(
            rf"^{leg}_tibia:\n  pos: .*$",
            f"{leg}_tibia:\n  pos: {_format_vec(new_tibia_pos)}",
            text,
            flags=re.M,
        )
        if n_subs != 1:
            raise ValueError(f"Expected exactly one {leg}_tibia entry with a pos")
    return text


if __name__ == "__main__":
    DST_PATH.write_text(make_rigging(SRC_PATH.read_text()))
    print(f"Wrote {DST_PATH}")
