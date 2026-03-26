import numpy as np
import warp as wp
import mujoco_warp as mjw

import flygym
from flygym.compose import Fly, ActuatorType, FlatGroundWorld
from flygym.anatomy import Skeleton, JointPreset, ActuatedDOFPreset, AxisOrder
from flygym.compose.pose import KinematicPose
from flygym.utils.math import Rotation3D

from flygym_examples.spotlight_data import MotionSnippet

snippet = MotionSnippet()
sim_timestep = 1e-4

joints_preset = JointPreset.LEGS_ONLY
actuated_dofs_preset = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
actuator_type = ActuatorType.POSITION
position_gain = 50.0
neutral_pose_file = flygym.assets_dir / "model/pose/neutral.yaml"
spawn_position = (0, 0, 0.7)  # xyz in mm
spawn_rotation = Rotation3D("quat", (1, 0, 0, 0))

fly = Fly()
axis_order = AxisOrder.YAW_PITCH_ROLL

# Add joints
skeleton = Skeleton(axis_order=axis_order, joint_preset=joints_preset)
neutral_pose = KinematicPose(path=neutral_pose_file)
fly.add_joints(skeleton, neutral_pose=neutral_pose)

# Add position actuators and set them to the neutral pose
actuated_dofs_list = fly.skeleton.get_actuated_dofs_from_preset(actuated_dofs_preset)
fly.add_actuators(
    actuated_dofs_list,
    actuator_type=actuator_type,
    kp=position_gain,
    neutral_input=neutral_pose,
)

# Add leg adhesion
fly.add_leg_adhesion()

# Add visuals
fly.colorize()
cam = fly.add_tracking_camera()

# Create a world and spawn the fly
world = FlatGroundWorld()
world.add_fly(fly, spawn_position, spawn_rotation)

mj_model, mj_data = world.compile()

# rc = mjw.create_render_context(
#     mj_model,
#     # cam_res=(256, 256),  # Override camera resolution (or per-camera list)
#     render_rgb=True,  # Enable RGB output (or per-camera list)
#     render_depth=True,  # Enable depth output (or per-camera list)
#     use_textures=True,  # Apply material textures
#     use_shadows=False,  # Enable shadow casting (slower)
#     enabled_geom_groups=[0, 1],  # Only render geoms in groups 0 and 1
#     cam_active=[True],  # Selectively enable/disable cameras
#     flex_render_smooth=True,  # Smooth shading for soft bodies
# )

from flygym.warp import GPUSimulation

n_worlds = 3000

sim = GPUSimulation(world, n_worlds)

playback_speed = 0.2
output_fps = 25
# tracked_worlds controls which parallel worlds have their frames stored.
# Only those world IDs can be passed to show_in_notebook / save_video.
# Keeping this small is critical: storing frames for all n_worlds worlds would
# require n_worlds × H × W × 4 bytes per frame (≈ 920 MB at n_worlds=3000),
# which would exhaust GPU memory. Tracking just 2 worlds costs ~600 KB per frame.
# renderer = sim.set_renderer(
#     cam,
#     playback_speed=playback_speed,
#     output_fps=output_fps,
#     worlds=[0],
# )

dof_angles = snippet.get_joint_angles(
    output_timestep=sim_timestep,
    output_dof_order=fly.get_actuated_jointdofs_order(actuator_type),
)
# dof_angles shape: (n_total_steps, n_dofs) — smoothed, interpolated, reordered
print(f"Total steps available: {dof_angles.shape[0]}, n_dofs: {dof_angles.shape[1]}")

sim_seconds = 1.0
sim_steps = int(sim_seconds / sim_timestep)

# Prepare the input array for all worlds
n_dofs = dof_angles.shape[1]
dof_angles_all_worlds = np.zeros((n_worlds, sim_steps, n_dofs), dtype=np.float32)

# World #0 will simulate steps [0, sim_steps),
# world #1 will simulate steps [1, sim_steps + 1), etc
for i in range(n_worlds):
    dof_angles_all_worlds[i] = dof_angles[i : i + sim_steps]

from tqdm import trange  # for progress bar


def run_simulation(
    sim,
    dof_angles_all_worlds,
    warmup_steps=500,
    fly_name=fly.name,
    actuator_type=actuator_type,
):
    n_worlds, sim_steps, n_dofs = dof_angles_all_worlds.shape
    sim.reset()
    sim.set_renderer(
        cam,
        playback_speed=playback_speed,
        output_fps=output_fps,
        worlds=[0],
    )
    # Turn adhesion on for all 6 legs across all worlds
    sim.set_leg_adhesion_states(fly_name, np.ones((n_worlds, 6), dtype=np.float32))
    sim.warmup()
    for step in trange(warmup_steps + sim_steps):
        control_inputs = dof_angles_all_worlds[:, step - warmup_steps, :]
        sim.set_actuator_inputs(fly_name, actuator_type, control_inputs)
        sim.step()
        sim.render_as_needed()

    sim.renderer.save_video(world_id=0, output_path="test_out.mp4")

dof_angles_all_worlds = dof_angles_all_worlds[:, :1000, :]
dof_angles_all_worlds_gpu = wp.array(dof_angles_all_worlds)
run_simulation(sim, dof_angles_all_worlds_gpu)
