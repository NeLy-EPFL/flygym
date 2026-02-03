from .anatomy import (
    # Anatomical features
    BodySegment,
    AnatomicalJoint,
    JointDOF,
    Skeleton,
    # Types for user config, along with TypeVars for typehints
    RotationAxis,
    RotationAxisLike,
    AxesSet,
    AxesSetLike,
    AxisOrder,
    AxisOrderLike,
    JointPreset,
    ContactBodiesPreset,
    # Constants
    SIDES,
    LEGS,
    BODY_POSITIONS,
    LEG_LINKS,
    ANTENNA_LINKS,
    PROBOSCIS_LINKS,
    ABDOMEN_LINKS,
    PASSIVE_TARSAL_LINKS,
    ALL_SEGMENT_NAMES,
    ALL_CONNECTED_SEGMENT_PAIRS,
)

from .model import Fly, ActuatorType
