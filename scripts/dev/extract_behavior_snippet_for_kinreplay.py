"""Extract snippets of behavior recording from published in

    Aymanns, F., Chen, C. L., & Ramdya, P. (2022). Descending neuron population dynamics
    during odor-evoked and spontaneous limb-dependent behaviors. Elife, 11, e81527.
    https://doi.org/10.7554/eLife.81527

for demonstrating kinematic replay.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from flygym import assets_dir
from flygym.anatomy import (
    Skeleton,
    AxisOrder,
    ActuatedDOFPreset,
    JointPreset,
    PASSIVE_TARSAL_LINKS,
)

DATA_FILE = Path(
    "~/projects/poseforge/bulk_data/kinematic_prior/aymanns2022/trials/BO_Gal4_fly1_trial001.pkl"
).expanduser()
FRAME_RANGE = (11000, 11501)
# FRAME_RANGE = (360, 561)
# FRAME_RANGE = (51650, 51950)
FPS = 100
CHILD_LINK_TO_AYMANNS_JOINT_NAME = {
    "coxa": "ThC",
    "trochanterfemur": "CTr",
    "tibia": "FTi",
    "tarsus1": "TiTa",
}
UNIT = "radian"
AXIS_ORDER = ("roll", "yaw", "pitch")
OUTPUT_PATH = assets_dir / "demo/aymans2022_behavior_clip.npz"

# Load and clip the data to the specified frame range.
df = pd.read_pickle(DATA_FILE)
df_clip = df.iloc[FRAME_RANGE[0] : FRAME_RANGE[1], :]

# Select and rename joints to match FlyGym's naming convention.
skeleton = Skeleton(
    axis_order=AxisOrder.ROLL_YAW_PITCH, joint_preset=JointPreset.LEGS_ONLY
)
joint_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
new_cols = {}
for joint_dof in joint_dofs:
    leg = joint_dof.child.pos
    joint_name = CHILD_LINK_TO_AYMANNS_JOINT_NAME[joint_dof.child.link]
    column_name = f"Angle__{leg.upper()}_leg_{joint_name}_{joint_dof.axis.value}"
    time_series = df_clip.loc[:, column_name]
    if joint_dof.child.pos[0] == "r" and joint_dof.axis.value in ("roll", "yaw"):
        # Flip sign for right leg's roll and yaw to match FlyGym's convention.
        time_series = -time_series
    new_cols[joint_dof.name] = time_series
df_clip = pd.DataFrame(new_cols)

# Save data and metadata as NPZ file
data = {
    "dof_angles": df_clip.values.astype(np.float16),
    "dof_order": list(df_clip.columns),
    "fps": FPS,
    "source_file": DATA_FILE.name,
    "frame_range": FRAME_RANGE,
    "unit": UNIT,
    "axis_order": AXIS_ORDER,
}
np.savez(OUTPUT_PATH, **data)
