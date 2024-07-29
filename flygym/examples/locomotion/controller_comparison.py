import subprocess
from itertools import chain, product
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dm_control.rl.control import PhysicsError
from joblib import Parallel, delayed
from matplotlib.image import AxesImage
from scipy.interpolate import interp1d
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from flygym import Camera, SingleFlySimulation
from flygym.arena import BlocksTerrain, FlatTerrain, GappedTerrain, MixedTerrain
from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps, ColorableFly
from flygym.examples.locomotion.rule_based_controller import (
    RuleBasedController,
    construct_rules_graph,
)
from flygym.preprogrammed import get_cpg_biases

plt.rcParams["font.family"] = "Arial"
plt.rcParams["pdf.fonttype"] = 42

########### SCRIPT PARAMS ############
terrains = ["flat", "gapped", "blocks", "mixed"]
controllers = ["cpg", "rule_based", "hybrid"]
controller_labels = ["CPG", "Rule-based", "Hybrid"]
palette = ["tab:blue", "tab:orange", "tab:brown"]

########### PATHS ############
outputs_dir = Path("outputs/controller_benchmark")
saves_dir = outputs_dir / "obs"
videos_dir = outputs_dir / "videos"
metadata_path = outputs_dir / "metadata.npz"

########### SIM PARAMS ############
env_seed = 0  # seed for randomizing spawn positions and block heights
n_trials = 20  # number of trials per (terrain, controller) pair
spawn_bbox = (-2, -2, 4, 4)  # (x-min, y-min, x-length, y-length)
spawn_z = 0.5  # z-coordinate of the fly at spawn
timestep = 1e-4  # simulation timestep
run_time = 1.5  # duration of each simulation run

########### CPG PARAMS ############
intrinsic_freqs = np.ones(6) * 12
intrinsic_amps = np.ones(6) * 1
phase_biases = get_cpg_biases("tripod")
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 20

########### RULE BASED PARAMS ############
rule_based_step_dur = 1 / np.mean(intrinsic_freqs)
weights = {
    "rule1": -10,
    "rule2_ipsi": 2.5,
    "rule2_contra": 1,
    "rule3_ipsi": 3.0,
    "rule3_contra": 2.0,
}
rules_graph = construct_rules_graph()

########### HYBRID PARAMS ############
correction_vectors = {
    # "leg pos": (Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1)
    # unit: radian
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}
right_leg_inversion = [1, -1, -1, 1, -1, 1, 1]
stumbling_force_threshold = -1
correction_rates = {"retraction": (800, 700), "stumbling": (2200, 1800)}
max_increment = 80
retraction_persistence = 20
persistence_init_thr = 20


########### FUNCTIONS ############
def save_obs_list(save_path, obs_list: list[dict]):
    """Save a list of observations to a compressed npz file."""
    array_dict = {}
    for k in obs_list[0]:
        array_dict[k] = np.array([i[k] for i in obs_list])
    np.savez_compressed(save_path, **array_dict)


def get_arena(arena: str):
    """Get the arena object based on the arena type."""
    if arena == "flat":
        return FlatTerrain()
    elif arena == "gapped":
        return GappedTerrain()
    elif arena == "blocks":
        # seed for randomized block heights
        return BlocksTerrain(rand_seed=env_seed)
    elif arena == "mixed":
        return MixedTerrain(rand_seed=env_seed)
    else:
        raise ValueError("Invalid arena type.")


def run_hybrid(
    sim: SingleFlySimulation,
    cpg_network: CPGNetwork,
    preprogrammed_steps: PreprogrammedSteps,
    run_time: float,
):
    """Run a simulation with hybrid controller.

    Parameters
    ----------
    sim : SingleFlySimulation
        Simulation object.
    cpg_network : CPGNetwork
        CPG network object.
    preprogrammed_steps : PreprogrammedSteps
        Preprogrammed steps object.
    run_time : float
        Duration of the simulation run.
    """
    step_phase_multiplier = {}
    increments = [0, 0.8, 0, -0.1, 0]

    for leg in preprogrammed_steps.legs:
        swing_start, swing_end = preprogrammed_steps.swing_period[leg]
        step_points = [
            swing_start,
            np.mean([swing_start, swing_end]),
            swing_end + np.pi / 4,
            np.mean([swing_end, 2 * np.pi]),
            2 * np.pi,
        ]
        preprogrammed_steps.swing_period[leg] = (swing_start, swing_end + np.pi / 4)
        step_phase_multiplier[leg] = interp1d(
            step_points, increments, fill_value="extrapolate"
        )

    retraction_correction = np.zeros(6)
    stumbling_correction = np.zeros(6)

    detected_segments = ["Tibia", "Tarsus1", "Tarsus2"]
    stumbling_sensors = {leg: [] for leg in preprogrammed_steps.legs}

    for j, sensor_name in enumerate(sim.fly.contact_sensor_placements):
        leg = sensor_name.split("/")[1][:2]  # sensor_name: eg. "Animat/LFTarsus1"
        segment = sensor_name.split("/")[1][2:]
        if segment in detected_segments:
            stumbling_sensors[leg].append(j)

    stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}

    obs, info = sim.reset()
    target_num_steps = int(run_time / sim.timestep)
    obs_list = []
    retraction_persistence_counter = np.zeros(6)

    for _ in range(target_num_steps):
        # retraction rule: does a leg need to be retracted from a hole?
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
            if retraction_correction[leg_to_correct_retraction] > persistence_init_thr:
                retraction_persistence_counter[leg_to_correct_retraction] = 1
        else:
            leg_to_correct_retraction = None

        # update persistence counter
        retraction_persistence_counter[retraction_persistence_counter > 0] += 1
        retraction_persistence_counter[
            retraction_persistence_counter > retraction_persistence
        ] = 0

        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []

        for j, leg in enumerate(preprogrammed_steps.legs):
            # update amount of retraction correction
            if (
                j == leg_to_correct_retraction or retraction_persistence_counter[j] > 0
            ):  # lift leg
                increment = correction_rates["retraction"][0] * sim.timestep
                retraction_correction[j] += increment
                sim.fly.change_segment_color(
                    sim.physics,
                    f"{leg}Tibia",
                    (1.0, 0.4117647058823529, 0.7058823529411765),
                )
            else:  # condition no longer met, lower leg
                decrement = correction_rates["retraction"][1] * sim.timestep
                retraction_correction[j] = max(0, retraction_correction[j] - decrement)
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", None)

            retract = retraction_correction[j] > 0

            # update amount of stumbling correction
            contact_forces = obs["contact_forces"][stumbling_sensors[leg], :]
            fly_orientation = obs["fly_orientation"]
            # force projection should be negative if against fly orientation
            force_proj = np.dot(contact_forces, fly_orientation)
            if (force_proj < stumbling_force_threshold).any():
                increment = correction_rates["stumbling"][0] * sim.timestep
                stumbling_correction[j] += increment
                if not retract:
                    sim.fly.change_segment_color(
                        sim.physics,
                        f"{leg}Tibia",
                        (0.11764705882352941, 0.5647058823529412, 1.0),
                    )
            else:
                decrement = correction_rates["stumbling"][1] * sim.timestep
                stumbling_correction[j] = max(0, stumbling_correction[j] - decrement)
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", None)

            # retraction correction is prioritized
            if retraction_correction[j] > 0:
                net_correction = retraction_correction[j]
                stumbling_correction[j] = 0
            else:
                net_correction = stumbling_correction[j]

            # get target angles from CPGs and apply correction
            my_joints_angles = preprogrammed_steps.get_joint_angles(
                leg, cpg_network.curr_phases[j], cpg_network.curr_magnitudes[j]
            )
            net_correction = np.clip(net_correction, 0, max_increment)
            if leg[0] == "R":
                net_correction *= right_leg_inversion[j]

            net_correction *= step_phase_multiplier[leg](
                cpg_network.curr_phases[j] % (2 * np.pi)
            )

            my_joints_angles += net_correction * correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
                leg, cpg_network.curr_phases[j]
            )

            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff),
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            obs_list.append(obs)
            sim.render()
        except PhysicsError:
            return obs_list, True

    return obs_list, False


def run_rule_based(
    sim: SingleFlySimulation,
    controller: RuleBasedController,
    run_time: float,
):
    """Run a simulation with rule based controller.

    Parameters
    ----------
    sim : SingleFlySimulation
        Simulation object.
    controller : RuleBasedController
        Rule based controller object.
    run_time : float
        Duration of the simulation run.
    """
    obs, info = sim.reset()
    obs_list = []
    for _ in range(int(run_time / sim.timestep)):
        controller.step()
        joint_angles = []
        adhesion_onoff = []
        for leg, phase in zip(controller.legs, controller.leg_phases):
            joint_angles_arr = controller.preprogrammed_steps.get_joint_angles(
                leg, phase
            )
            joint_angles.append(joint_angles_arr.flatten())
            adhesion_onoff.append(
                controller.preprogrammed_steps.get_adhesion_onoff(leg, phase)
            )
        action = {
            "joints": np.concatenate(joint_angles),
            "adhesion": np.array(adhesion_onoff),
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            obs_list.append(obs)
            sim.render()
        except PhysicsError:
            return obs_list, True

    return obs_list, False


def run_cpg(
    sim: SingleFlySimulation,
    cpg_network: CPGNetwork,
    preprogrammed_steps: PreprogrammedSteps,
    run_time: float,
):
    """Run a simulation with CPG controller.

    Parameters
    ----------
    sim : SingleFlySimulation
        Simulation object.
    cpg_network : CPGNetwork
        CPG network object.
    preprogrammed_steps : PreprogrammedSteps
        Preprogrammed steps object.
    run_time : float
        Duration of the simulation run.
    """
    obs, info = sim.reset()
    obs_list = []
    for _ in range(int(run_time / sim.timestep)):
        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(preprogrammed_steps.legs):
            my_joints_angles = preprogrammed_steps.get_joint_angles(
                leg, cpg_network.curr_phases[i], cpg_network.curr_magnitudes[i]
            )
            joints_angles.append(my_joints_angles)
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
                leg, cpg_network.curr_phases[i]
            )
            adhesion_onoff.append(my_adhesion_onoff)
        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            sim.render()
            obs_list.append(obs)
        except PhysicsError:
            return obs_list, True

    return obs_list, False


def run_all(arena: str, seed: int, pos: np.ndarray, verbose: bool = False):
    """Run experiments for all controllers in a given arena and seed.

    Parameters
    ----------
    arena : str
        Arena type.
    seed : int
        Random seed.
    pos : np.ndarray
        Spawn position.
    verbose : bool, optional
        Print experiment results, by default False.
    """
    save_paths = {c: saves_dir / f"{c}_{arena}_{seed}.npz" for c in controllers}
    video_paths = {c: videos_dir / f"{c}_{arena}_{seed}.mp4" for c in controllers}

    if all(p.exists() for p in chain(save_paths.values(), video_paths.values())):
        return
    else:
        pass

    preprogrammed_steps = PreprogrammedSteps()
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in preprogrammed_steps.legs
        for segment in ["Tibia"] + [f"Tarsus{i}" for i in range(1, 6)]
    ]

    # Initialize the simulation
    fly = ColorableFly(
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose="stretch",
        control="position",
        spawn_pos=pos,
        contact_sensor_placements=contact_sensor_placements,
        actuator_forcerange=(-65.0, 65.0),
    )
    terrain = get_arena(arena)
    cam = Camera(fly=fly, play_speed=0.1, camera_id="Animat/camera_right")
    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=terrain,
    )

    # run cpg simulation
    sim.reset()
    cpg_network = CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
        seed=seed,
    )
    cpg_network.reset()
    obs_list, phys_error = run_cpg(sim, cpg_network, preprogrammed_steps, run_time)
    displacements = obs_list[-1]["fly"][0] - obs_list[0]["fly"][0]

    if verbose:
        print(f"CPG experiment {seed}: {displacements}", end="")
        print(" ended with physics error" if phys_error else "")

    cam.save_video(video_paths["cpg"], 0)
    save_obs_list(save_paths["cpg"], obs_list)

    # run rule based simulation
    preprogrammed_steps.duration = rule_based_step_dur
    controller = RuleBasedController(
        timestep=timestep,
        rules_graph=rules_graph,
        weights=weights,
        preprogrammed_steps=preprogrammed_steps,
        seed=seed,
    )
    sim.reset()

    obs_list, phys_error = run_rule_based(sim, controller, run_time)
    displacements = obs_list[-1]["fly"][0] - obs_list[0]["fly"][0]

    if verbose:
        print(f"Rule based experiment {seed}: {displacements}", end="")
        print(" ended with physics error" if phys_error else "")

    cam.save_video(video_paths["rule_based"], 0)
    save_obs_list(save_paths["rule_based"], obs_list)

    # run hybrid simulation
    np.random.seed(seed)
    sim.reset()
    cpg_network.random_state = np.random.RandomState(seed)
    cpg_network.reset()
    obs_list, phys_error = run_hybrid(sim, cpg_network, preprogrammed_steps, run_time)
    displacements = obs_list[-1]["fly"][0] - obs_list[0]["fly"][0]

    if verbose:
        print(f"Hybrid experiment {seed}: {displacements}", end="")
        print(" ended with physics error" if phys_error else "")

    cam.save_video(video_paths["hybrid"], 0)
    save_obs_list(save_paths["hybrid"], obs_list)


########### RUN SIMULATIONS ############

# Create directories
for d in [saves_dir, videos_dir, metadata_path.parent]:
    d.mkdir(parents=True, exist_ok=True)

# Generate random positions
rng = np.random.RandomState(env_seed)
positions = rng.rand(n_trials, 2) * spawn_bbox[2:] + spawn_bbox[:2]
positions = np.column_stack((positions, np.full(n_trials, spawn_z)))

# Save metadata to yaml
np.savez_compressed(metadata_path, run_time=run_time, positions=positions)

# Run experiments
it = [(a, s, p) for a, (s, p) in product(terrains, enumerate(positions))]
Parallel(n_jobs=-1)(delayed(run_all)(*i, True) for i in tqdm(it))


########### VIDEO GENERATION ############
def get_video_reader(trial_id, controller, terrain):
    """Get a video reader for a given trial, controller and terrain."""
    path = videos_dir / f"{controller}_{terrain}_{trial_id}.mp4"
    return imageio.get_reader(path)


def get_video_props():
    """Get the video properties (width, height, fps)."""
    with get_video_reader(0, "cpg", "flat") as reader:
        metadata = reader.get_meta_data()
        return (metadata["size"], metadata["fps"])


im_size, fps = get_video_props()
empty_img = np.zeros((*im_size[::-1], 3), dtype=np.uint8)


def init_fig(trial_id: int) -> list[AxesImage]:
    """Initialize the figure for the video comparison.

    Parameters
    ----------
    trial_id : int
        The trial id.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    images : list[matplotlib.image.AxesImage]
        The image objects.
    """
    fig, axs = plt.subplots(len(controllers), len(terrains), figsize=(20, 11.5))
    fig.subplots_adjust(
        left=0.1, right=0.95, top=0.9, bottom=0.05, hspace=0.03, wspace=0.03
    )

    images = np.empty((len(controllers), len(terrains)), object)

    for i, j in product(range(len(controllers)), range(len(terrains))):
        ax = axs[i, j]
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        images[i, j] = ax.imshow(empty_img)
        if i + j == 0:
            ax.text(7, 60, f"Trial {trial_id + 1}", fontsize=20, fontname="Arial")

    ylabels = [f"{i}\ncontroller" for i in controller_labels]

    for i, (text, color) in enumerate(zip(ylabels, palette)):
        axs[i, 0].set_ylabel(text, color=color, size=26, fontname="Arial", labelpad=10)

    terrain_labels = [f"{i.capitalize()} terrain" for i in terrains]

    for j, text in enumerate(terrain_labels):
        axs[0, j].set_title(text, size=26, fontname="Arial")

    axs[-1, -1].text(
        0.57, -0.1, "0.1x speed", transform=ax.transAxes, size=26, fontname="Arial"
    )
    return fig, images.ravel()


def write_trial_video(trial_id: int, n_frames=450):
    it = list(product(controllers, terrains))
    video_readers = [get_video_reader(trial_id, c, t) for c, t in it]
    fig, images = init_fig(trial_id)

    with imageio.get_writer(outputs_dir / f"{trial_id:02d}.mp4", fps=fps) as writer:
        for frame_id in range(n_frames):
            if frame_id < 5:
                continue
            for image, video_reader in zip(images, video_readers):
                if frame_id < len(video_reader):
                    video_reader.set_image_index(frame_id)
                    img = video_reader.get_next_data()[36:]
                else:
                    img = empty_img
                image.set_data(img)

            fig.canvas.draw()
            frame = np.array(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)

    for reader in video_readers:
        reader.close()


Parallel(n_jobs=-1)(delayed(write_trial_video)(i) for i in range(n_trials))

with open(outputs_dir / "video_list.txt", "w") as f:
    f.writelines([f"file {i:02d}.mp4\n" for i in range(n_trials)])

subprocess.run(
    [
        *"ffmpeg -y -f concat -safe 0 -i".split(),
        (outputs_dir / "video_list.txt").as_posix(),
        "-c",
        "copy",
        (outputs_dir / "controller_comparison.mp4").as_posix(),
    ]
)

for i in range(n_trials):
    Path(outputs_dir / f"{i:02d}.mp4").unlink()

(outputs_dir / "video_list.txt").unlink()


########### PLOTTING ############
def get_data():
    rows = []
    for controller, terrain in product(controllers, terrains):
        for npz_file in list(saves_dir.glob(f"{controller}_{terrain}_*.npz")):
            fly_pos = np.load(npz_file)["fly"]
            dist = np.linalg.norm(np.diff(fly_pos[[0, -1], 0, :2].T))
            speed = dist / (len(fly_pos) * timestep)
            rows.append([controller.lower(), terrain.lower(), speed])
    return pd.DataFrame(rows, columns=["controller", "terrain", "speed"]).dropna()


def get_pvals():
    pvals = {}
    rows = []

    for terrain in terrains:
        x_hybrid = df.query("controller == 'hybrid' and terrain == @terrain")["speed"]
        rows.append([])
        for controller in ["cpg", "rule_based"]:
            x = df.query("controller == @controller and terrain == @terrain")["speed"]
            u, p = mannwhitneyu(x, x_hybrid, alternative="less", method="asymptotic")
            pvals[terrain, controller] = p
            rows[-1].append(f"U = {int(u)}, p = {p:.1E}")

    print(pd.DataFrame(rows, terrains, ["cpg < hybrid", "rule-based < hybrid"]))
    return pvals


def plot_comparison(df: pd.DataFrame):
    fig = plt.figure(figsize=(6, 2))
    widths = np.array([1, 0.2, 1, 0.2, 1, 0.2, 1])
    widths /= widths.sum()
    axs = {}
    props = {
        "boxprops": {"fc": "None", "ec": "k", "lw": 0.5},
        "medianprops": {"c": "gray", "lw": 0.5},
        "whiskerprops": {"c": "gray", "lw": 0.5},
        "capprops": {"c": "gray", "lw": 0.5},
    }

    for j, terrain in enumerate(terrains):
        axs[terrain] = ax = fig.add_axes([widths[: j * 2].sum(), 0, widths[j * 2], 1])
        ax.set_title(terrain.capitalize() + " terrain")
        data = df[df["terrain"].eq(terrain)]
        kw = dict(x="controller", y="speed", data=data, ax=ax)
        sns.boxplot(width=0.6, showfliers=False, **props, **kw)
        sns.swarmplot(hue="controller", size=1.75, palette=palette, legend=False, **kw)
        ax.set_xmargin(0.1)
        ax.xaxis.set_visible(False)
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        if j:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.sharey(axs[terrains[j - 1]])
        else:
            ax.set_ylabel("Avg. speed (mm/s)")
            ax.set_ylim(0, 15)
            ax.set_yticks(np.arange(0, 16, 5))

    handles = [
        plt.Line2D([0], [0], marker="o", ms=5, label=s, c=c)
        for s, c in zip(controller_labels, palette)
    ]
    ax = fig.add_axes([widths[:2].sum(), 0, widths[2:].sum(), 1])
    ax.axis("off")
    fig.legend(
        handles=handles,
        labels=controller_labels,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False,
        labelcolor=palette,
        handletextpad=0.6,
        handlelength=0,
        fontsize="large",
    )
    return axs


def get_asterisks(p):
    return ["***", "**", "*", "ns"][np.digitize(p, [1e-3, 1e-2, 5e-2])]


df = get_data()
pvals = get_pvals()
axs = plot_comparison(df)
axs["gapped"].set_ylim(0, None)

for (terrain, ax), (j, controller) in product(axs.items(), enumerate(controllers[:2])):
    x, y = (j + 2) / 2, (j * 0.8 - 1.2) / 10
    kw = dict(x=x, y=y, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.errorbar(xerr=(2 - j) / 2, c="k", lw=0.5, capsize=2, capthick=0.5, **kw)
    s = get_asterisks(pvals[terrain, controller])
    ax.text(s=s, ha="center", va="bottom" if s == "ns" else "center", size=8, **kw)

plt.savefig(outputs_dir / "speed_comparison.pdf", bbox_inches="tight", transparent=True)
