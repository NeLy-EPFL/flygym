import numpy as np
from tqdm import trange

from flygym.mujoco import Parameters, NeuroMechFly
from flygym.mujoco.arena import MixedTerrain
from flygym.mujoco.examples.common import PreprogrammedSteps
from flygym.mujoco.examples.cpg_controller import CPGNetwork


def _find_stumbling_sensor_indices(
    nmf, detected_segments=["Tibia", "Tarsus1", "Tarsus2"]
):
    stumbling_sensors = {leg: [] for leg in preprogrammed_steps.legs}
    for i, sensor_name in enumerate(nmf.contact_sensor_placements):
        leg = sensor_name.split("/")[1][:2]  # sensor_name: eg. "Animat/LFTarsus1"
        segment = sensor_name.split("/")[1][2:]
        if segment in detected_segments:
            stumbling_sensors[leg].append(i)
    stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}
    if any(v.size != len(detected_segments) for v in stumbling_sensors.values()):
        raise RuntimeError(
            "Contact detection must be enabled for all tibia, tarsus1, and tarsus2 "
            "segments for stumbling detection."
        )
    return stumbling_sensors


def run_hybrid_simulation(
    nmf,
    cpg_network,
    preprogrammed_steps,
    correction_vectors,
    correction_rates,
    stumbling_force_threshold,
    run_time,
):
    # find the indices contact force sensors for each leg - need for stumbling detection
    stumbling_sensors = _find_stumbling_sensor_indices(nmf)

    retraction_correction = np.zeros(6)
    stumbling_correction = np.zeros(6)

    obs, info = nmf.reset()
    for _ in trange(int(run_time / nmf.sim_params.timestep)):
        # retraction rule: is any leg stuck in a hole and needs to be retracted?
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
        else:
            leg_to_correct_retraction = None

        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(preprogrammed_steps.legs):
            # update retraction correction amounts
            if i == leg_to_correct_retraction:  # lift leg
                increment = correction_rates["retraction"][0] * nmf.timestep
                retraction_correction[i] += increment
                nmf.change_segment_color(f"{leg}Tibia", (1, 0, 0, 1))
            else:  # condition no longer met, lower leg
                decrement = correction_rates["retraction"][1] * nmf.timestep
                retraction_correction[i] = max(0, retraction_correction[i] - decrement)
                nmf.change_segment_color(f"{leg}Tibia", (0.5, 0.5, 0.5, 1))

            # update stumbling correction amounts
            contact_forces = obs["contact_forces"][stumbling_sensors[leg], :]
            fly_orientation = obs["fly_orientation"]
            # force projection should be negative if against fly orientation
            force_proj = np.dot(contact_forces, fly_orientation)
            if (force_proj < stumbling_force_threshold).any():
                increment = correction_rates["stumbling"][0] * nmf.timestep
                stumbling_correction[i] += increment
                nmf.change_segment_color(f"{leg}Femur", (1, 0, 0, 1))
            else:
                decrement = correction_rates["stumbling"][1] * nmf.timestep
                stumbling_correction[i] = max(0, stumbling_correction[i] - decrement)
                nmf.change_segment_color(f"{leg}Femur", (0.5, 0.5, 0.5, 1))

            # retraction correction has priority
            if retraction_correction[i] > 0:
                net_correction = retraction_correction[i]
            else:
                net_correction = stumbling_correction[i]

            # get target angles from CPGs and apply correction
            my_joints_angles = preprogrammed_steps.get_joint_angles(
                leg, cpg_network.curr_phases[i], cpg_network.curr_magnitudes[i]
            )
            my_joints_angles += net_correction * correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
                leg, cpg_network.curr_phases[i]
            )
            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        obs, reward, terminated, truncated, info = nmf.step(action)
        nmf.render()


if __name__ == "__main__":
    run_time = 1
    timestep = 1e-4

    # Define leg raise correction vectors
    correction_vectors = {
        # "leg pos": (Coxa, Coxa_roll, Coxa_yaw, Femur, Fimur_roll, Tibia, Tarsus1)
        "F": np.array([0, 0, 0, -0.02, 0, 0.016, 0]),
        "M": np.array([-0.015, 0, 0, 0.004, 0, 0.01, -0.008]),
        "H": np.array([0, 0, 0, -0.01, 0, 0.005, 0]),
    }

    # Define leg raise rates
    correction_rates = {"retraction": (500, 1000 / 3), "stumbling": (2000, 500)}

    # Initialize CPG network
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
    cpg_network = CPGNetwork(
        timestep=1e-4,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
    )

    # Initialize preprogrammed steps
    preprogrammed_steps = PreprogrammedSteps()

    # Initialize NeuroMechFly simulation
    sim_params = Parameters(
        timestep=1e-4,
        render_mode="saved",
        render_playspeed=0.1,
        enable_adhesion=True,
        draw_adhesion=True,
    )
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in preprogrammed_steps.legs
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    arena = MixedTerrain()
    nmf = NeuroMechFly(
        sim_params=sim_params,
        arena=arena,
        init_pose="stretch",
        control="position",
        spawn_pos=(0, 0, 0.2),
        contact_sensor_placements=contact_sensor_placements,
    )

    # Run simulation
    run_hybrid_simulation(
        nmf=nmf,
        cpg_network=cpg_network,
        preprogrammed_steps=preprogrammed_steps,
        correction_vectors=correction_vectors,
        correction_rates=correction_rates,
        stumbling_force_threshold=-1,
        run_time=run_time,
    )

    # Save video
    nmf.save_video("./outputs/hybrid_controller.mp4")
