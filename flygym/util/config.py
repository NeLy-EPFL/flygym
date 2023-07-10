from typing import List


# DoF definitions
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


# Geometries
all_tarsi_links = [
    f"{side}{pos}Tarsus{i}" for side in "LR" for pos in "FMH" for i in range(1, 6)
]

colors = {
    "wings": [91 / 100, 96 / 100, 97 / 100, 0.3],
    "eyes": [67 / 100, 21 / 100, 12 / 100, 1],
    "body": [160 / 255, 120 / 255, 50 / 255, 1],
    "coxa": [
        [150 / 255, 100 / 255, 30 / 255],
        [0, 0, 0],
        [1, 1, 1, 0.8],
    ],
    "femur": [
        [160 / 255, 110 / 255, 40 / 255],
        [0, 0, 0],
        [1, 1, 1, 0.7],
    ],
    "tibia": [
        [170 / 255, 120 / 255, 50 / 255],
        [0, 0, 0],
        [1, 1, 1, 0.6],
    ],
    "tarsus": [
        [190 / 255, 130 / 255, 60 / 255],
        [0, 0, 0],
        [1, 1, 1, 0.5],
    ],
    "A12345": [
        [130 / 255, 80 / 255, 10 / 255],
        [210 / 255, 170 / 255, 120 / 255],
        [178 / 255, 126 / 255, 51 / 255],
    ],
    "thorax": [
        [150 / 255, 100 / 255, 30 / 255],
        [178 / 255, 126 / 255, 51 / 255],
        [1, 1, 1, 1],
    ],
    "A6": [
        [100 / 255, 50 / 255, 0 / 255],
        [210 / 255, 170 / 255, 120 / 255],
        [178 / 255, 126 / 255, 51 / 255],
    ],
    "proboscis": [150 / 255, 100 / 255, 30 / 255, 1],
    "aristas": [67 / 255, 52 / 255, 40 / 255, 255 / 255],
    "antennas": [150 / 255, 100 / 255, 30 / 255, 1],
    "halteres": [150 / 255, 110 / 255, 60 / 255, 255 / 255],
}


def get_collision_geoms(config: str = "all") -> List[str]:
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
