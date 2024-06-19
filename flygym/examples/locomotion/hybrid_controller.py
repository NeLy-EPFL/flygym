import numpy as np
from tqdm import trange
from scipy.interpolate import interp1d

from flygym.examples.locomotion import CPGNetwork
from flygym import Fly, Camera, SingleFlySimulation
from flygym.examples.locomotion import PreprogrammedSteps
from flygym.arena import MixedTerrain
from dm_control.rl.control import PhysicsError


intrinsic_freqs = np.ones(6) * 12
intrinsic_amps = np.ones(6) * 1
phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 20

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
max_increment = 80 / 1e-4
retraction_persistence = 20 / 1e-4
persistence_init_thr = 20 / 1e-4


def run_hybrid_simulation(sim, cpg_network, preprogrammed_steps, run_time):
    step_phase_multiplier = {}

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
        increment_vals = [0, 0.8, 0, -0.1, 0]

        step_phase_multiplier[leg] = interp1d(
            step_points, increment_vals, kind="linear", fill_value="extrapolate"
        )

    retraction_correction = np.zeros(6)
    stumbling_correction = np.zeros(6)

    detected_segments = ["Tibia", "Tarsus1", "Tarsus2"]
    stumbling_sensors = {leg: [] for leg in preprogrammed_steps.legs}
    for i, sensor_name in enumerate(sim.fly.contact_sensor_placements):
        leg = sensor_name.split("/")[1][:2]  # sensor_name: eg. "Animat/LFTarsus1"
        segment = sensor_name.split("/")[1][2:]
        if segment in detected_segments:
            stumbling_sensors[leg].append(i)
    stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}

    obs, info = sim.reset()

    target_num_steps = int(run_time / sim.timestep)
    obs_hist = []
    info_hist = []

    retraction_persistence_counter = np.zeros(6)

    retraction_persistence_counter_hist = np.zeros((6, target_num_steps))

    for k in trange(target_num_steps):
        # retraction rule: does a leg need to be retracted from a hole?
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
            if (
                retraction_correction[leg_to_correct_retraction]
                > persistence_init_thr * sim.timestep
            ):
                retraction_persistence_counter[leg_to_correct_retraction] = 1
        else:
            leg_to_correct_retraction = None

        # update persistence counter
        retraction_persistence_counter[retraction_persistence_counter > 0] += 1
        retraction_persistence_counter[
            retraction_persistence_counter > retraction_persistence * sim.timestep
        ] = 0
        retraction_persistence_counter_hist[:, k] = retraction_persistence_counter

        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []

        all_net_corrections = np.zeros(6)

        for i, leg in enumerate(preprogrammed_steps.legs):
            # update amount of retraction correction
            if (
                i == leg_to_correct_retraction or retraction_persistence_counter[i] > 0
            ):  # lift leg
                increment = correction_rates["retraction"][0] * sim.timestep
                retraction_correction[i] += increment
                sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (1, 0, 0, 1))
            else:  # condition no longer met, lower leg
                decrement = correction_rates["retraction"][1] * sim.timestep
                retraction_correction[i] = max(0, retraction_correction[i] - decrement)
                sim.fly.change_segment_color(
                    sim.physics, f"{leg}Tibia", (0.5, 0.5, 0.5, 1)
                )

            # update amount of stumbling correction
            contact_forces = obs["contact_forces"][stumbling_sensors[leg], :]
            fly_orientation = obs["fly_orientation"]
            # force projection should be negative if against fly orientation
            force_proj = np.dot(contact_forces, fly_orientation)
            if (force_proj < stumbling_force_threshold).any():
                increment = correction_rates["stumbling"][0] * sim.timestep
                stumbling_correction[i] += increment
                sim.fly.change_segment_color(sim.physics, f"{leg}Femur", (1, 0, 0, 1))
            else:
                decrement = correction_rates["stumbling"][1] * sim.timestep
                stumbling_correction[i] = max(0, stumbling_correction[i] - decrement)
                if retraction_correction[i] <= 0:
                    sim.fly.change_segment_color(
                        sim.physics, f"{leg}Femur", (0.5, 0.5, 0.5, 1)
                    )

            # retraction correction is prioritized
            if retraction_correction[i] > 0:
                net_correction = retraction_correction[i]
                stumbling_correction[i] = 0
            else:
                net_correction = stumbling_correction[i]

            # get target angles from CPGs and apply correction
            my_joints_angles = preprogrammed_steps.get_joint_angles(
                leg, cpg_network.curr_phases[i], cpg_network.curr_magnitudes[i]
            )
            net_correction = np.clip(net_correction, 0, max_increment * sim.timestep)
            if leg[0] == "R":
                net_correction *= right_leg_inversion[i]

            net_correction *= step_phase_multiplier[leg](
                cpg_network.curr_phases[i] % (2 * np.pi)
            )

            my_joints_angles += net_correction * correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
                leg, cpg_network.curr_phases[i]
            )

            all_net_corrections[i] = net_correction

            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff),
        }

        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            info["net_corrections"] = np.array(all_net_corrections)
            obs_hist.append(obs)
            info_hist.append(info)

            sim.render()
        except PhysicsError:
            print("Simulation was interrupted because of a physics error")
            return obs_hist, info_hist, True

    return obs_hist, info_hist, False


if __name__ == "__main__":
    run_time = 1.0
    timestep = 1e-4

    preprogrammed_steps = PreprogrammedSteps()

    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in preprogrammed_steps.legs
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    # Initialize the simulation
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose="stretch",
        control="position",
        contact_sensor_placements=contact_sensor_placements,
    )
    cam = Camera(fly=fly, play_speed=0.1)

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=MixedTerrain(),
    )

    # run cpg simulation
    obs, inf = sim.reset()
    cpg_network = CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
    )

    print(f"Spawning fly at {obs['fly'][0]} mm")

    obs_list, inf_list, had_physics_error = run_hybrid_simulation(
        sim, cpg_network, preprogrammed_steps, run_time
    )
    print(f"Simulation terminated: {obs_list[-1]['fly'][0] - obs_list[0]['fly'][0]}")

    # Save video
    cam.save_video(f"./outputs/hybrid_controller.mp4", 0)
