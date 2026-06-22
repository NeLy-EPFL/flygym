"""Smoke test: replay experimentally recorded fly walking on the CPU.

This is a script version of the core of
``tutorials/2_replaying_experimental_recordings.ipynb``. It loads a snippet of
experimentally recorded kinematics (collected with the Spotlight system), replays
it on a single CPU-simulated fly using position actuators, and records the
resulting joint angles, actuator torques, and leg-joint site positions.

It is intended as a human-friendly end-to-end test and a profiling target: run it
without arguments to just exercise the pipeline (and print a performance report),
or pass ``--save-data DIR`` to additionally write the observation history, plots,
and rendered video to ``DIR``.

Example:
    uv run python scripts/dev/run_cpu_smoketest.py --save-data outputs/cpu_smoketest
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display needed
import matplotlib.pyplot as plt
from tqdm import trange

from flygym import Simulation
from flygym.compose import (
    NeuroMechFly,
    KinematicPosePreset,
    ActuatorType,
    FlatGroundWorld,
)
from flygym.anatomy import (
    Skeleton,
    AxisOrder,
    JointPreset,
    ActuatedDOFPreset,
    JointDOF,
    RotationAxis,
    BodySegment,
)
from flygym.utils.math import Rotation3D
from flygym_demo.spotlight_data import MotionSnippet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to write the observation history, plots, and rendered "
        "video. Errors out if it already exists. If omitted, nothing is saved.",
    )
    parser.add_argument(
        "--sim-duration-sec",
        type=float,
        default=None,
        help="Duration to simulate, in seconds. Defaults to the full recording.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1e-4,
        help="Simulation timestep in seconds (default: 1e-4).",
    )
    parser.add_argument(
        "--actuator-gain",
        type=float,
        default=150.0,
        help="Position actuator gain in uN*mm/rad (default: 150).",
    )
    return parser.parse_args()


def build_model(actuator_gain: float):
    """Build the fly, world, and simulation, mirroring tutorial 2."""
    axis_order = AxisOrder.YAW_PITCH_ROLL
    articulated_joints = JointPreset.LEGS_ONLY
    actuated_dofs = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    neutral_pose = KinematicPosePreset.NEUTRAL
    actuator_type = ActuatorType.POSITION

    fly = NeuroMechFly()
    skeleton = Skeleton(axis_order=axis_order, joint_preset=articulated_joints)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)

    actuated_dofs_list = fly.skeleton.get_actuated_dofs_from_preset(actuated_dofs)
    fly.add_actuators(
        actuated_dofs_list,
        actuator_type=actuator_type,
        kp=actuator_gain,
        neutral_input=neutral_pose,
    )

    sites = fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    tracking_cam = fly.add_tracking_camera()
    fly.add_leg_adhesion()

    spawn_pos = [0, 0, 0.7]  # center of thorax 0.7 mm above the ground
    spawn_rot = Rotation3D(format="quat", values=[1, 0, 0, 0])  # no rotation

    world = FlatGroundWorld()
    world.add_fly(fly, spawn_pos, spawn_rot)

    sim = Simulation(world)
    sim.set_renderer(tracking_cam)
    return fly, sim, sites, actuator_type


def plot_joint_angle_tracking(
    snippet, fly, actuator_type, timegrid, target_angles, simulated_angles, leg="lf"
):
    """Compare target vs. achieved joint angles for one leg (tutorial 2 cell 26)."""
    fig, axes = plt.subplots(7, 1, figsize=(9, 6.5), tight_layout=True, sharex=True)
    for dof_idx, (parent_link, child_link, axis) in enumerate(snippet.dofs_per_leg):
        ax = axes[dof_idx]
        parent_name = "c_thorax" if parent_link == "thorax" else f"{leg}_{parent_link}"
        child_name = f"{leg}_{child_link}"
        target_dof = JointDOF(
            BodySegment(parent_name), BodySegment(child_name), RotationAxis(axis)
        )

        actuated_idx = fly.get_actuated_jointdofs_order(actuator_type).index(target_dof)
        ts_target = np.rad2deg(target_angles[:, actuated_idx])
        ax.plot(timegrid, ts_target, label="Target", linestyle=":", color="C0")

        dof_idx_all = fly.get_jointdofs_order().index(target_dof)
        ts_sim = np.rad2deg(simulated_angles[:, dof_idx_all])
        ax.plot(timegrid, ts_sim, label="Simulated", color="C1")

        ax.set_ylabel("Angle\n(°)")
        ax.set_title(f"{parent_link}-{child_link} {axis}", fontsize="medium")
        if dof_idx == 6:
            ax.set_xlabel("Time (s)")
        if dof_idx == 0:
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    fig.suptitle(f"{leg.upper()} leg joint angles: target vs. simulated")
    return fig


def plot_pose_over_time(site_positions, timestep):
    """Plot fly leg pose snapshots over time in 3D (tutorial 2 cell 28)."""
    nsteps_sim = site_positions.shape[0]
    snapshot_interval = int(0.3 / timestep)  # a snapshot every 0.3 s
    n_snapshots = max(nsteps_sim // snapshot_interval, 1)
    n_anatomical_joints_per_leg = 8

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    for snapshot_idx in range(n_snapshots):
        step_idx = snapshot_idx * snapshot_interval
        for leg_idx in range(6):
            slice_ = slice(
                leg_idx * n_anatomical_joints_per_leg,
                (leg_idx + 1) * n_anatomical_joints_per_leg,
            )
            x = site_positions[step_idx, slice_, 0]
            y = site_positions[step_idx, slice_, 1]
            z = site_positions[step_idx, slice_, 2]
            alpha = (snapshot_idx / n_snapshots) * 0.8 + 0.2
            ax.plot(x, y, z, color="black", alpha=alpha)
        leadline_origin = (x[0], y[0], z[0] + 1)
        leadline_end = (x[0], y[0], z[0] + 3)
        text_center = (x[0], y[0], z[0] + 3.2)
        ax.plot(*zip(leadline_origin, leadline_end), color="black", alpha=0.5)
        t = step_idx * timestep
        ax.text(*text_center, f"t={t:.1f}s", color="black", ha="center", va="bottom")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_aspect("equal")
    fig.suptitle("Leg-joint pose over time")
    return fig


def main() -> None:
    args = parse_args()

    data_dir: Path | None = args.save_data
    if data_dir is not None:
        if data_dir.exists():
            sys.exit(f"Error: output directory already exists: {data_dir}")
        data_dir.mkdir(parents=True)

    snippet = MotionSnippet()
    fly, sim, sites, actuator_type = build_model(args.actuator_gain)
    fly_name = fly.name

    # Build the target angle sequence (smoothed, upsampled, reordered).
    target_angles = snippet.get_joint_angles(
        output_timestep=args.timestep,
        output_dof_order=fly.get_actuated_jointdofs_order(actuator_type),
    )
    full_duration_sec = snippet.joint_angles.shape[0] / snippet.data_fps
    duration_sec = args.sim_duration_sec or full_duration_sec
    timegrid = np.arange(0, duration_sec, args.timestep)
    nsteps_sim = min(timegrid.size, target_angles.shape[0])
    timegrid = timegrid[:nsteps_sim]

    n_dofs = len(fly.get_jointdofs_order())
    n_actuated_dofs = len(fly.get_actuated_jointdofs_order(actuator_type))
    simulated_joint_angles = np.full((nsteps_sim, n_dofs), np.nan, dtype=np.float32)
    actuator_torques = np.full((nsteps_sim, n_actuated_dofs), np.nan, dtype=np.float32)
    site_positions = np.full((nsteps_sim, len(sites), 3), np.nan, dtype=np.float32)

    print(f"Simulating {nsteps_sim} steps ({nsteps_sim * args.timestep:.3f} s)...")
    sim.reset()
    sim.set_leg_adhesion_states(fly_name, np.ones(6, dtype=bool))
    sim.warmup()

    for step_idx in trange(nsteps_sim, desc="Simulating"):
        sim.set_actuator_inputs(fly_name, actuator_type, target_angles[step_idx, :])
        sim.step_with_profile()
        simulated_joint_angles[step_idx, :] = sim.get_joint_angles(fly_name)
        actuator_torques[step_idx, :] = sim.get_actuator_forces(fly_name, actuator_type)
        site_positions[step_idx, :, :] = sim.get_site_positions(fly_name)
        sim.render_as_needed_with_profile()

    sim.print_performance_report()

    if data_dir is None:
        return

    print(f"Saving observation history, plots, and video to {data_dir}...")
    np.savez_compressed(
        data_dir / "observations.npz",
        timegrid=timegrid,
        target_angles=target_angles[:nsteps_sim],
        simulated_joint_angles=simulated_joint_angles,
        actuator_torques=actuator_torques,
        site_positions=site_positions,
        jointdofs_order=np.array(
            [str(d) for d in fly.get_jointdofs_order()], dtype=object
        ),
        actuated_jointdofs_order=np.array(
            [str(d) for d in fly.get_actuated_jointdofs_order(actuator_type)],
            dtype=object,
        ),
        sites_order=np.array([str(s) for s in fly.get_sites_order()], dtype=object),
    )

    fig = plot_joint_angle_tracking(
        snippet,
        fly,
        actuator_type,
        timegrid,
        target_angles[:nsteps_sim],
        simulated_joint_angles,
    )
    fig.savefig(data_dir / "joint_angle_tracking.png", dpi=120)
    plt.close(fig)

    fig = plot_pose_over_time(site_positions, args.timestep)
    fig.savefig(data_dir / "pose_over_time.png", dpi=120)
    plt.close(fig)

    sim.renderer.save_video(data_dir / "replay.mp4")
    print("Done.")


if __name__ == "__main__":
    main()
