import numpy as np
from typing import List
import numpy as np


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

cuticule_color = np.array([150 / 255, 100 / 255, 30 / 255])
cuticule_variation = np.array([178 / 255, 126 / 255, 51 / 255])
cuticule_bellow = np.array([210 / 255, 170 / 255, 120 / 255])
cuticule_A6_top = np.array([100 / 255, 50 / 255, 0 / 255])

colors = {
    "wings": [80 / 100, 80 / 100, 90 / 100, 0.3],
    "eyes": [67 / 100, 21 / 100, 12 / 100, 1],
    "body": [160 / 255, 120 / 255, 50 / 255, 1],
    "head": [
        cuticule_color,
        [178 / 255, 126 / 255, 51 / 255],
        [1, 1, 1, 1],
    ],
    "coxa": [
        cuticule_color,
        [0, 0, 0],
        [1, 1, 1, 0.8],
    ],
    "femur": [
        cuticule_color + 10 / 255,
        [0, 0, 0],
        [1, 1, 1, 0.7],
    ],
    "tibia": [
        cuticule_color + 20 / 255,
        [0, 0, 0],
        [1, 1, 1, 0.6],
    ],
    "tarsus": [
        cuticule_color + 30 / 255,
        [0, 0, 0],
        [1, 1, 1, 0.5],
    ],
    "A12345": [
        cuticule_color,
        cuticule_bellow,
        cuticule_variation,
    ],
    "thorax": [
        cuticule_color,
        cuticule_variation,
        [1, 1, 1, 1],
    ],
    "A6": [
        cuticule_A6_top,
        cuticule_bellow,
        cuticule_variation,
    ],
    "proboscis": [
        cuticule_color,
        [0, 0, 0],
        [1, 1, 1, 0.8],
    ],
    "aristas": [67 / 255, 52 / 255, 40 / 255, 255 / 255],
    "antennas": [
        cuticule_color,
        [0, 0, 0],
        [1, 1, 1, 0.8],
    ],
    "halteres": [150 / 255, 110 / 255, 60 / 255, 0.6],
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


# Vision
# fovx_per_eye = 146.71
# fovy_per_eye = fovx_per_eye * (2 / np.sqrt(3))
fovy_per_eye = 157  # 160 5.5 3.25
raw_img_height_px = 512
raw_img_width_px = 450
retina_side_len_hex = 16
num_ommatidia_per_eye = 3 * retina_side_len_hex**2 - 3 * retina_side_len_hex + 1
fisheye_distortion_coefficient = 3.8
fisheye_zoom = 2.72
hidden_segments_for_vision = [
    "LFCoxa",
    "LEye",
    "LArista",
    "LFuniculus",
    "LPedicel",
    "RFCoxa",
    "REye",
    "RArista",
    "RFuniculus",
    "RPedicel",
    "Head",
    "Rostrum",
    "Haustellum",
    "Thorax",
]


# Leg adhesion
# joint velocities threshold extracted from experiments
adhesion_speed_thresholds = np.array(
    [
        -22.24454997,
        -12.13565398,
        -9.14855537,
        -20.7181815,
        12.49711737,
        10.15158114,
    ]
)


# Sensor positions
# fmt: off
sensor_positions = {
    # sensor name: (parent, position rel parent, orientation, marker rgba)
    "LEye_cam": ("LEye", (-0.03, 0.38, 0), (1.57, -0.4676, 0.08), (0.07, 0.45, 0.35, 1)),
    "REye_cam": ("REye", (-0.03, -0.38, 0), (-1.57, -0.47, 3.06), (0.07, 0.45, 0.35, 1)),
    "LMaxillaryPalp_sensor": ("Rostrum", (-0.15, 0.15, -0.15), (0.9, 0.73, 0.08, 1)),
    "RMaxillaryPalp_sensor": ("Rostrum", (-0.15, -0.15, -0.15), (0.9, 0.73, 0.08, 1)),
    "LAntenna_sensor": ("LFuniculus", (0.02, 0.00, -0.10), (0.08, 0.4, 0.9, 1)),
    "RAntenna_sensor": ("RFuniculus", (0.02, 0.00, -0.10), (0.08, 0.4, 0.9, 1)),
}
# fmt: on
