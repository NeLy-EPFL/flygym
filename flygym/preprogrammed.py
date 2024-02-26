from typing import List
from flygym.state import KinematicPose
from flygym.util import get_data_path


all_leg_dofs = [
    f"joint_{side}{pos}{dof}"
    for side in "LR"
    for pos in "FMH"
    for dof in [
        "Coxa",
        "Coxa_roll",
        "Coxa_yaw",
        "Femur",
        "Femur_roll",
        "Tibia",
        "Tarsus1",
    ]
]

leg_dofs_3_per_leg = [
    f"joint_{side}{pos}{dof}"
    for side in "LR"
    for pos in "FMH"
    for dof in ["Coxa" if pos == "F" else "Coxa_roll", "Femur", "Tibia"]
]

all_tarsi_links = [
    f"{side}{pos}Tarsus{i}" for side in "LR" for pos in "FMH" for i in range(1, 6)
]


def get_preprogrammed_pose(pose: str) -> KinematicPose:
    """Load the preprogrammed pose given the key. Available poses are found
    in the data directory: ``flygym/data/pose/pose_{key}.yaml``. Included
    poses are:

    - "stretch": all legs are fully extended sideways to ensure that the
      legs are not embedded into the ground upon initialization, which
      breaks the physics. This is the suggested initial pose.
    - "tripod": a snapshot of a tethered fly walking in a tripod gait.
    - "zero": the zero pose of the NeuroMechFly model. Not that the fly
      should be spawned significantly above the ground to ensure that the
      legs are not extended into the ground upon initialization.
    """
    data_path = get_data_path("flygym", "data") / "pose" / f"pose_{pose}.yaml"
    if not data_path.is_file():
        raise ValueError(f"Pose {pose} does not exist.")
    return KinematicPose.from_yaml(data_path)


def get_collision_geometries(config: str = "all") -> List[str]:
    """Get the list of collision geometries given the key: "all" (all body
    segments), "legs-no-coxa" (all leg segments excluding the coxa
    segments), "tarsi" (all tarsus segments), and "none" (nothing).
    """
    if config == "legs":
        return [
            f"{side}{pos}{dof}_collision"
            for side in "LR"
            for pos in "FMH"
            for dof in [
                "Coxa",
                "Femur",
                "Tibia",
                "Tarsus1",
                "Tarsus2",
                "Tarsus3",
                "Tarsus4",
                "Tarsus5",
            ]
        ]
    elif config == "legs-no-coxa":
        return [
            f"{side}{pos}{dof}_collision"
            for side in "LR"
            for pos in "FMH"
            for dof in [
                "Femur",
                "Tibia",
                "Tarsus1",
                "Tarsus2",
                "Tarsus3",
                "Tarsus4",
                "Tarsus5",
            ]
        ]
    elif config == "tarsi":
        return [
            f"{side}{pos}{dof}_collision"
            for side in "LR"
            for pos in "FMH"
            for dof in ["Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
        ]
    elif config == "none":
        return []
    else:
        raise ValueError(f"Unknown collision geometry configuration: {config}")
