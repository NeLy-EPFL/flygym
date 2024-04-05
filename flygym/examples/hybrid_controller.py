import numpy as np
from flygym.examples.cpg_controller import CPGNetwork

from flygym import Fly, Camera, SingleFlySimulation
from flygym.examples import PreprogrammedSteps
from flygym.arena import FlatTerrain, GappedTerrain, BlocksTerrain, MixedTerrain
from dm_control.rl.control import PhysicsError

from tqdm import trange
import pickle

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

correction_rates = {"retraction": (800, 700), "stumbling": (2200, 2100)}
max_increment = 80
retraction_persistance = 20
persistance_init_thr = 20


def run_hybrid_simulation(sim, cpg_network, preprogrammed_steps, run_time):

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
    print(obs["fly"][0])

    target_num_steps = int(run_time / sim.timestep)
    obs_list = []

    retraction_perisitance_counter = np.zeros(6)

    physics_error = False

    for k in trange(target_num_steps):
        # retraction rule: does a leg need to be retracted from a hole?
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.06:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
            if retraction_correction[leg_to_correct_retraction] > persistance_init_thr:
                retraction_perisitance_counter[leg_to_correct_retraction] = 1
        else:
            leg_to_correct_retraction = None

        # update persistance counter
        retraction_perisitance_counter[retraction_perisitance_counter > 0] += 1
        retraction_perisitance_counter[
            retraction_perisitance_counter > retraction_persistance
        ] = 0

        cpg_network.step()
        joints_angles = []
        adhesion_onoff = []

        all_net_corrections = []

        for i, leg in enumerate(preprogrammed_steps.legs):
            # update amount of retraction correction
            if (
                i == leg_to_correct_retraction or retraction_perisitance_counter[i] > 0
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
            net_correction = np.clip(net_correction, 0, max_increment)
            if leg[0] == "R":
                net_correction *= right_leg_inversion[i]

            my_joints_angles += net_correction * correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            all_net_corrections.append(net_correction)

            # get adhesion on/off signal
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
                leg, cpg_network.curr_phases[i]
            )
            # No adhesion in stumbling or retracted
            is_trembling = (force_proj < stumbling_force_threshold).any()
            is_retracting = i == leg_to_correct_retraction
            is_retracting |= retraction_perisitance_counter[i] > 0
            my_adhesion_onoff *= np.logical_not(is_trembling or is_retracting)
            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            obs["net_correction"] = all_net_corrections
            obs_list.append(obs)
            sim.render()
        except PhysicsError:
            physics_error = True
            print("Simulation was interupted because of a physics error")
            break

    return obs_list, physics_error


if __name__ == "__main__":

    run_time = 0.5
    timestep = 1e-4

    preprogrammed_steps = PreprogrammedSteps()

    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in preprogrammed_steps.legs
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    np.random.seed(0)

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=contact_sensor_placements,
    )

    cam = Camera(fly=fly, play_speed=0.1, camera_id="Animat/camera_right")
    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=1e-4,
        arena=GappedTerrain(),
    )

    cpg_network = CPGNetwork(
        timestep=1e-4,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
        seed=0,
    )

    obs_list, had_physics_error = run_hybrid_simulation(
        sim, cpg_network, preprogrammed_steps, run_time
    )

    x_pos = obs_list[-1]["fly"][0][0]
    print(f"Final x position: {x_pos:.4f} mm")

    # save all joint angles
    joint_angles = [obs["joints"][0] for obs in obs_list]
    with open("./outputs/hybrid_controller_joint_angles.pkl", "wb") as f:
        pickle.dump(obs_list, f)

    # Save video
    cam.save_video("./outputs/hybrid_controller.mp4", 0)
