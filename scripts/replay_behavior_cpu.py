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

Pass ``--profile PATH`` to record a sampling profile with
`py-spy <https://github.com/benfred/py-spy>`_ and write it in speedscope format to
``PATH``; open the result at https://speedscope.app (or with the ``speedscope``
CLI). py-spy is attached to the *already-running* process for the duration of the
simulation loop only -- the one-time imports and model building, which otherwise
dominate the flame graph as a tall ``_find_and_load`` / ``exec_module`` tower, are
never sampled, so what you see is the loop itself. Sampling uses ``--native``, so
native (C/C++) frames -- notably MuJoCo's physics step, which dominates this
workload (``mj_step`` -> ``mj_projectConstraint``, ``mju_cholFactorNumeric``, ...) --
show up alongside the Python frames. On Linux this needs no sudo (we nominate py-spy
as an allowed tracer via ``prctl(PR_SET_PTRACER)``); a longer ``--sim-duration-sec``
simply yields more samples.

Pass ``--mujoco-timing`` for a complementary, symbolication-free view: it installs
MuJoCo's internal timer callback (``mjcb_time``) so the C engine fills in
``mjData.timer`` per phase, then prints a per-phase breakdown of the physics step
(kinematics, collision broad/narrow-phase, constraint solve, integration, ...)
after the run. This is the natural way to see *where inside* the physics step the
time goes, which stripped/inlined native stacks make hard to read. It adds a
per-phase Python callback overhead, so read it as a relative breakdown rather than
an absolute-throughput measurement.

Example:
    uv run python scripts/replay_behavior_cpu.py --save-data outputs/cpu_smoketest
    uv run python scripts/replay_behavior_cpu.py --profile outputs/cpu.speedscope.json
    uv run python scripts/replay_behavior_cpu.py --mujoco-timing
"""

import os
import sys
import time
import shutil
import signal
import ctypes
import argparse
import threading
import subprocess
from pathlib import Path
from contextlib import contextmanager, nullcontext

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display needed
import matplotlib.pyplot as plt
from tqdm import trange
from tabulate import tabulate

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
        "video. Overwritten if it already exists. If omitted, nothing is saved.",
    )
    parser.add_argument(
        "--sim-duration-sec",
        type=float,
        default=None,
        help="Duration to simulate, in seconds. Defaults to half the recording.",
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
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        metavar="PATH",
        help="Record a sampling profile of the simulation loop with py-spy and write "
        "it in speedscope format to PATH (e.g. profile.speedscope.json), then open it "
        "at https://speedscope.app. Attaches `py-spy record --native` to the loop only "
        "(imports/model build are excluded). Requires the 'dev' extra.",
    )
    parser.add_argument(
        "--mujoco-timing",
        action="store_true",
        help="Print MuJoCo's built-in per-phase timing breakdown of the physics step "
        "(via the mjcb_time callback) after the run. Adds per-step overhead, so read "
        "it as a relative breakdown rather than an absolute-throughput measurement.",
    )
    return parser.parse_args()


# py-spy sampling rate (Hz). A physics step is only tens of microseconds, so we
# sample well above py-spy's 100 Hz default.
_PYSPY_RATE = 500


def ensure_pyspy() -> None:
    """Exit with a helpful message if the py-spy executable is not on PATH."""
    if shutil.which("py-spy") is None:
        sys.exit(
            "Error: --profile requires py-spy, which was not found on PATH. "
            "Install the 'dev' extra (e.g. `uv sync --extra dev`)."
        )


@contextmanager
def pyspy_attached(output_path: Path):
    """Attach py-spy to *this* process for the duration of the wrapped block only.

    Unlike launching the whole script under py-spy, this profiles just the simulation
    loop: the one-time imports and model building -- which otherwise dominate and
    clutter the flame graph -- are never sampled.

    py-spy runs as a separate process that ptraces us. Under Linux's default Yama
    policy (``ptrace_scope=1``) that requires us to nominate it as an allowed tracer
    via ``prctl(PR_SET_PTRACER)``; we pass ``PR_SET_PTRACER_ANY`` so no sudo is needed.
    On stop we send SIGINT, which makes py-spy flush the speedscope file while this
    process keeps running (e.g. to go on and save data).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sys.platform == "linux":
        # PR_SET_PTRACER == 0x59616d61 ("Yama"); PR_SET_PTRACER_ANY == (unsigned long)-1.
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(0x59616D61, ctypes.c_ulong(0xFFFFFFFFFFFFFFFF), 0, 0, 0)
        except OSError:
            pass  # best effort: attach can still succeed if ptrace_scope == 0

    cmd = [
        "py-spy",
        "record",
        "--pid",
        str(os.getpid()),
        "--native",
        "--rate",
        str(_PYSPY_RATE),
        "--format",
        "speedscope",
        "--output",
        str(output_path),
    ]
    print("Attaching py-spy to the simulation loop:\n  " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    ready = threading.Event()
    output_lines: list[str] = []

    def _reader() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            output_lines.append(line.rstrip())
            if "Sampling" in line:  # py-spy prints this once it begins sampling
                ready.set()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # Wait until py-spy is actually sampling (or has died trying to attach).
    start = time.monotonic()
    while time.monotonic() - start < 60 and not ready.is_set():
        if proc.poll() is not None:
            break
        ready.wait(timeout=0.2)
    if not ready.is_set():
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        sys.exit(
            "Error: py-spy did not start sampling:\n  " + "\n  ".join(output_lines)
        )

    try:
        yield
    finally:
        proc.send_signal(signal.SIGINT)  # tell py-spy to flush the speedscope file
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        reader_thread.join(timeout=5)
        for line in output_lines:
            if any(k in line for k in ("Wrote", "Error", "error")):
                print(line)


# MuJoCo's `mjData.timer` phases, in the order we report them, with an indent depth
# for a readable nested layout. Times are reported relative to `mjTIMER_STEP`.
# (Imported lazily inside the helpers so a normal run never imports mujoco directly.)
_MUJOCO_TIMER_PHASES = [
    ("mjTIMER_STEP", "step (total)", 0),
    ("mjTIMER_FORWARD", "forward dynamics", 1),
    ("mjTIMER_POSITION", "position", 2),
    ("mjTIMER_POS_KINEMATICS", "kinematics", 3),
    ("mjTIMER_POS_INERTIA", "inertia", 3),
    ("mjTIMER_POS_COLLISION", "collision", 3),
    ("mjTIMER_COL_BROAD", "broad-phase", 4),
    ("mjTIMER_COL_NARROW", "narrow-phase", 4),
    ("mjTIMER_POS_MAKE", "make constraints", 3),
    ("mjTIMER_POS_PROJECT", "project constraints", 3),
    ("mjTIMER_VELOCITY", "velocity", 2),
    ("mjTIMER_ACTUATION", "actuation", 2),
    ("mjTIMER_CONSTRAINT", "constraint solve", 1),
    ("mjTIMER_ADVANCE", "integrate (advance)", 1),
]


def enable_mujoco_timing() -> None:
    """Install MuJoCo's timer callback so the C engine fills in ``mjData.timer``.

    ``mjData.timer`` always counts how often each phase runs, but its durations stay
    zero unless a ``mjcb_time`` callback is registered; ``time.perf_counter`` (in
    seconds) is the timestamp source MuJoCo's own ``simulate`` viewer uses.
    """
    import mujoco

    mujoco.set_mjcb_time(time.perf_counter)


def reset_mujoco_timers(mj_data) -> None:
    """Zero ``mjData.timer`` so the report excludes warm-up / pre-loop steps."""
    for i in range(len(mj_data.timer)):
        mj_data.timer[i].number = 0
        mj_data.timer[i].duration = 0.0


def print_mujoco_timing(mj_data) -> None:
    """Print MuJoCo's built-in per-phase timing breakdown of the physics step.

    Reads ``mjData.timer`` (populated by `enable_mujoco_timing`), MuJoCo's own C-level
    instrumentation -- a symbolication-free view of where time goes *inside* a step,
    complementary to the py-spy flame graph. All times are over the step count and
    relative to the total ``mjTIMER_STEP`` time.
    """
    import mujoco

    n_steps = mj_data.timer[int(mujoco.mjtTimer.mjTIMER_STEP)].number
    step_total_s = mj_data.timer[int(mujoco.mjtTimer.mjTIMER_STEP)].duration
    if n_steps == 0 or step_total_s == 0.0:
        print("No MuJoCo timing recorded (did the loop run any steps?).")
        return

    rows = []
    for enum_name, label, depth in _MUJOCO_TIMER_PHASES:
        timer = mj_data.timer[int(getattr(mujoco.mjtTimer, enum_name))]
        if timer.number == 0:
            continue
        rows.append(
            [
                "  " * depth + label,
                1e3 * timer.duration,  # total ms
                1e6 * timer.duration / n_steps,  # us per step
                100 * timer.duration / step_total_s,  # % of step
            ]
        )
    print(f"\nMuJoCo per-phase physics timing ({n_steps} steps):")
    print(
        tabulate(
            rows,
            headers=["Phase", "Total (ms)", "us/step", "% of step"],
            floatfmt=".3f",
            tablefmt="rounded_outline",
        )
    )


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

    if args.profile is not None:
        ensure_pyspy()  # fail fast, before the (untimed) model build

    data_dir: Path | None = args.save_data
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)  # overwrite if it exists

    snippet = MotionSnippet()
    fly, sim, sites, actuator_type = build_model(args.actuator_gain)
    fly_name = fly.name

    if args.mujoco_timing:
        enable_mujoco_timing()

    # Build the target angle sequence (smoothed, upsampled, reordered).
    target_angles = snippet.get_joint_angles(
        output_timestep=args.timestep,
        output_dof_order=fly.get_actuated_jointdofs_order(actuator_type),
    )
    full_duration_sec = snippet.joint_angles.shape[0] / snippet.data_fps
    duration_sec = args.sim_duration_sec or full_duration_sec / 2
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

    # Zero the MuJoCo timers after warm-up so the report covers only the timed loop.
    if args.mujoco_timing:
        reset_mujoco_timers(sim.mj_data)

    # Attach py-spy (if profiling) around the loop only -- imports/model build above
    # are deliberately excluded. Time the loop as a whole (no per-step instrumentation,
    # which would otherwise add Python frames to the sampling profile).
    profile_cm = (
        pyspy_attached(args.profile) if args.profile is not None else nullcontext()
    )
    with profile_cm:
        loop_start_ns = time.perf_counter_ns()
        for step_idx in trange(nsteps_sim, desc="Simulating"):
            sim.set_actuator_inputs(fly_name, actuator_type, target_angles[step_idx, :])
            sim.step()
            simulated_joint_angles[step_idx, :] = sim.get_joint_angles(fly_name)
            actuator_torques[step_idx, :] = sim.get_actuator_forces(
                fly_name, actuator_type
            )
            site_positions[step_idx, :, :] = sim.get_site_positions(fly_name)
            sim.render_as_needed()
        loop_walltime_s = (time.perf_counter_ns() - loop_start_ns) / 1e9

    throughput = nsteps_sim / loop_walltime_s
    print(
        f"Simulated {nsteps_sim} steps in {loop_walltime_s:.3f}s "
        f"({throughput:.0f} steps/s, {throughput * args.timestep:.2f}x realtime, "
        f"end-to-end incl. observation recording and rendering)."
    )
    if args.mujoco_timing:
        print_mujoco_timing(sim.mj_data)

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
