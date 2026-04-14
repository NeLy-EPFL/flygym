"""Public flybody-specific API.

This module re-exports flybody anatomy types so users can import from
``flygym.flybody`` instead of deep asset paths.
"""

from flygym.assets.model.flybody.anatomy_flybody import (
    FlybodyRotationAxis,
    WingFlybodyRotationAxis,
    FlybodyAxesSet,
    WingFlybodyAxesSet,
    FlybodyAxisOrder,
    WingFlybodyAxisOrder,
    FlybodyBodySegment,
    FlybodyJointPreset,
    FlybodyActuatedDOFPreset,
    FlybodyContactBodiesPreset,
    FlybodySkeleton,
    FlybodyJointDOF,
)

__all__ = [
    "FlybodyRotationAxis",
    "WingFlybodyRotationAxis",
    "FlybodyAxesSet",
    "WingFlybodyAxesSet",
    "FlybodyAxisOrder",
    "WingFlybodyAxisOrder",
    "FlybodyBodySegment",
    "FlybodyJointPreset",
    "FlybodyActuatedDOFPreset",
    "FlybodyContactBodiesPreset",
    "FlybodySkeleton",
    "FlybodyJointDOF",
]
