"""Locations of bundled musculoskeletal-model assets."""

from flygym import assets_dir

MUSCULOSKELETAL_DIR = assets_dir / "musculoskeletal"
"""Directory holding FlyMimic's musculoskeletal MJCF + meshes."""

DEFAULT_MUSCLE_XML = MUSCULOSKELETAL_DIR / "best_combined_arm_damping_stiff_cvt3.xml"
"""FlyMimic's muscle-driven fly with passive joint stiffness + spring refs.

This is the ``arm_damping_stiff`` variant: same 15 left-front-leg muscles and
body geometry as the base muscle model, plus biologically-motivated passive
joint stiffness (0.4) and per-joint spring reference angles.
"""
