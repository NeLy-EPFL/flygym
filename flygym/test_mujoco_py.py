import time

import mujoco
import mujoco.viewer

from flygym.fly import Fly
from flygym.arena import Tethered

import flygym.util as util
from flygym.util import get_data_path

from dm_control.mjcf import export_with_assets
from pathlib import Path

config = util.load_config()

fly = Fly(init_pose="tripod")
arena = Tethered()
arena.spawn_entity(fly.model, fly.spawn_pos, fly.spawn_orientation)

xml_with_assets_path = Path("./fly_model")
xml_with_assets_path.mkdir(exist_ok=True, parents=True)

#export model with assets
file_name = "cool_fly.xml"
export_with_assets(arena.root_element, xml_with_assets_path, out_file_name=file_name)

m = mujoco.MjModel.from_xml_path(str(xml_with_assets_path/file_name))
d = mujoco.MjData(m)

mujoco.GLContext(1000, 1000)

default_pose = []
"""for leg in preprogrammed_steps.legs:
    default_pose.append(preprogrammed_steps.get_joint_angles(leg, 0.0))"""

with mujoco.viewer.launch_passive(m, d) as viewer:
  
  #viewer.cam.fixedcamid = "Animat/camera_left"

  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:

    viewer._cam = mujoco.MjvCamera()
    viewer._cam.azimuth = 180.0

    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)