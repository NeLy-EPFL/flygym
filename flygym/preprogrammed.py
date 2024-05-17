import numpy as np
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
    if config == "all":
        # fmt: off
        return [
            "Thorax", "A1A2", "A3", "A4", "A5", "A6", "Head_roll", "Head_yaw",
            "Head", "LEye", "LPedicel_roll", "LPedicel_yaw", "LPedicel",
            "LFuniculus_roll", "LFuniculus_yaw", "LFuniculus", "LArista_roll",
            "LArista_yaw", "LArista", "REye", "Rostrum", "Haustellum",
            "RPedicel_roll", "RPedicel_yaw", "RPedicel", "RFuniculus_roll",
            "RFuniculus_yaw", "RFuniculus", "RArista_roll", "RArista_yaw",
            "RArista", "LFCoxa_roll", "LFCoxa_yaw", "LFCoxa", "LFFemur",
            "LFFemur_roll", "LFTibia", "LFTarsus1", "LFTarsus2", "LFTarsus3",
            "LFTarsus4", "LFTarsus5", "LHaltere_roll", "LHaltere_yaw",
            "LHaltere", "LHCoxa_roll", "LHCoxa_yaw", "LHCoxa", "LHFemur",
            "LHFemur_roll", "LHTibia", "LHTarsus1", "LHTarsus2", "LHTarsus3",
            "LHTarsus4", "LHTarsus5", "LMCoxa_roll", "LMCoxa_yaw", "LMCoxa",
            "LMFemur", "LMFemur_roll", "LMTibia", "LMTarsus1", "LMTarsus2",
            "LMTarsus3", "LMTarsus4", "LMTarsus5", "LWing_roll", "LWing_yaw",
            "LWing", "RFCoxa_roll", "RFCoxa_yaw", "RFCoxa", "RFFemur",
            "RFFemur_roll", "RFTibia", "RFTarsus1", "RFTarsus2", "RFTarsus3",
            "RFTarsus4", "RFTarsus5", "RHaltere_roll", "RHaltere_yaw",
            "RHaltere", "RHCoxa_roll", "RHCoxa_yaw", "RHCoxa", "RHFemur",
            "RHFemur_roll", "RHTibia", "RHTarsus1", "RHTarsus2", "RHTarsus3",
            "RHTarsus4", "RHTarsus5", "RMCoxa_roll", "RMCoxa_yaw", "RMCoxa",
            "RMFemur", "RMFemur_roll", "RMTibia", "RMTarsus1", "RMTarsus2",
            "RMTarsus3", "RMTarsus4", "RMTarsus5", "RWing_roll", "RWing_yaw",
            "RWing"
        ]
        # fmt: on
    elif config == "legs":
        return [
            f"{side}{pos}{dof}"
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
            f"{side}{pos}{dof}"
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
            f"{side}{pos}{dof}"
            for side in "LR"
            for pos in "FMH"
            for dof in ["Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
        ]
    elif config == "none":
        return []
    else:
        raise ValueError(f"Unknown collision geometry configuration: {config}")


def get_cpg_biases(gait: str) -> np.ndarray:
    """Define CPG biases for different gaits."""
    if gait.lower() == "tripod":
        phase_biases = np.array(
            [
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        phase_biases *= np.pi

    elif gait.lower() == "tetrapod":
        phase_biases = np.array(
            [
                [0, 1, 2, 2, 0, 1],
                [2, 0, 1, 1, 2, 0],
                [1, 2, 0, 0, 1, 2],
                [1, 2, 0, 0, 1, 2],
                [0, 1, 2, 2, 0, 1],
                [2, 0, 1, 1, 2, 0],
            ],
            dtype=np.float64,
        )
        phase_biases *= 2 * np.pi / 3

    elif gait.lower() == "wave":
        phase_biases = np.array(
            [
                [0, 1, 2, 3, 4, 5],
                [5, 0, 1, 2, 3, 4],
                [4, 5, 0, 1, 2, 3],
                [3, 4, 5, 0, 1, 2],
                [2, 3, 4, 5, 0, 1],
                [1, 2, 3, 4, 5, 0],
            ],
            dtype=np.float64,
        )
        phase_biases *= 2 * np.pi / 6

    else:
        raise ValueError(f"Unknown gait: {gait}")

    return phase_biases
