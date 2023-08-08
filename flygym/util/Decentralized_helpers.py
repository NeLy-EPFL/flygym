import numpy as np

######## Initialisation ########


def define_swing_stance_starts(
    nmf, data_block, use_adhesion=False, n_steps_stabil=1000
):
    leg_stepping_order = ["LF", "RF", "LH", "RH", "LM", "RM"]

    interp_step_duration = data_block.shape[1]

    nmf.reset()
    n_joints = len(nmf.actuated_joints)
    joint_ids = np.arange(n_joints).astype(int)
    # Map the id of the joint to the leg it belongs to (usefull to go through the steps for each legs)
    match_leg_to_joints = np.array(
        [
            i
            for joint in nmf.actuated_joints
            for i, leg in enumerate(leg_stepping_order)
            if leg in joint
        ]
    )

    # Map the id of the end effector (Tarsus5) to the leg it belongs to
    leg_to_end_effector_id = {
        leg: i
        for i, end_effector in enumerate(nmf.last_tarsalseg_names)
        for leg in leg_stepping_order
        if leg in end_effector
        if leg in end_effector
    }

    # Number of timesteps between each (fly) step
    n_rest_timesteps = 2000

    # Map the id of the force sensors to the leg it belongs to
    leg_force_sensors_ids = {leg: [] for leg in leg_stepping_order}
    for i, collision_geom in enumerate(nmf.contact_sensor_placements):
        for leg in leg_stepping_order:
            if collision_geom.startswith("Animat/" + leg + "Tarsus"):
                leg_force_sensors_ids[leg].append(i)

    # Record the touch sensor data for each leg for each timepoint
    touch_sensor_data = np.zeros(
        (len(leg_stepping_order), interp_step_duration + n_rest_timesteps - 1)
    )

    # Get the position of the last segment of the tarsus for each leg in the
    leg_tarsi_pos_id = {
        leg: [i]
        for leg in leg_stepping_order
        for i, joint in enumerate(nmf.actuated_joints)
        if leg in joint and "Tarsus1" in joint
    }
    position_data = np.zeros(
        (len(leg_stepping_order), interp_step_duration + n_rest_timesteps - 1, 3)
    )

    # Run the simulation until the fly is stable
    for k in range(n_steps_stabil):
        action = {"joints": data_block[joint_ids, 0], "adhesion": np.zeros(6)}
        obs, info, _, _, _ = nmf.step(action)

    # Lets step each leg on after the other collect touch sensor data as well as 3d coordinates of the last segment of the tarsus
    for i, leg in enumerate(leg_stepping_order):
        # "Boolean" like indexer for the stepping leg
        joints_to_actuate = np.zeros(len(nmf.actuated_joints)).astype(int)
        joints_to_actuate[match_leg_to_joints == i] = 1

        for k in range(interp_step_duration):
            # Advance the stepping in the joints of the stepping leg
            joint_pos = data_block[joint_ids, joints_to_actuate * k]

            if use_adhesion:
                adhesion = nmf.get_adhesion_vector()
            else:
                adhesion = np.zeros(6)

            action = {"joints": joint_pos, "adhesion": adhesion}
            obs, info, _, _, _ = nmf.step(action)
            # Get the touch sensor data from physics (sum of the Tarsus bellonging to a leg)
            touch_sensor_data[i, k] = np.sum(
                obs["contact_forces"][2, leg_force_sensors_ids[leg]]
            )
            # Get the position data from physics
            position_data[i, k, :] = obs["end_effectors"].reshape(
                len(leg_stepping_order), 3
            )[
                leg_to_end_effector_id[leg]
            ]  # Get the position data from physics

        # Rest between steps
        for j in range(n_rest_timesteps):
            action = {"joints": data_block[joint_ids, 0], "adhesion": np.zeros(6)}
            obs, info, _, _, _ = nmf.step(action)
            touch_sensor_data[i, k + j] = np.sum(
                obs["contact_forces"][2, leg_force_sensors_ids[leg]]
            )
            position_data[i, k + j, :] = obs["end_effectors"].reshape(
                len(leg_stepping_order), 3
            )[
                leg_to_end_effector_id[leg]
            ]  # Get the position data from physics

    leg_swing_starts = {}
    leg_stance_starts = {}

    stride = 50  # Number of timesteps to check for contact
    eps = 1.5  # Threshold for detecting contact

    for i, leg in enumerate(leg_stepping_order):
        # Plot contact forces
        k = 0
        # Until you find a swing onset keep going (as long as k is less than the length of the data)
        while k < len(touch_sensor_data[i]) and not np.all(
            touch_sensor_data[i, k : k + stride] < eps
        ):
            k += 1
        swing_start = k
        if k < len(touch_sensor_data[i]):
            # Find the first time the contact force is above the threshold
            try:
                stance_start = (
                    np.where(touch_sensor_data[i, swing_start:] > eps + 0.5)[0][0]
                    + swing_start
                )
            except IndexError:
                # If could not find a stance starts, set it to 0.03s
                stance_start_time = 0.03
                stance_start = int(stance_start_time / nmf.timestep)
            leg_swing_starts[leg] = swing_start
            leg_stance_starts[leg] = stance_start
        else:
            leg_swing_starts[leg] = 0
            leg_stance_starts[leg] = 0

    return leg_swing_starts, leg_stance_starts, position_data, touch_sensor_data


######## Running variables ########


def update_stepping_advancement(stepping_advancement, legs, interp_step_duration):
    # Advance the stepping advancement of each leg that are stepping, reset the advancement of the legs that are done stepping
    for k, leg in enumerate(legs):
        if stepping_advancement[k] >= interp_step_duration - 1:
            stepping_advancement[k] = 0
        elif stepping_advancement[k] > 0:
            stepping_advancement[k] += 1
    return stepping_advancement


def compute_leg_scores(
    rule1_corresponding_legs,
    rule1_weight,
    rule2_corresponding_legs,
    rule2_weight,
    rule2_weight_contralateral,
    rule3_corresponding_legs,
    rule3_weight,
    rule3_weight_contralateral,
    stepping_advancement,
    leg_corresp_id,
    leg_stance_starts,
    interp_step_duration,
    legs,
):
    rule1_contrib = np.zeros(len(legs))
    rule2_contrib = np.zeros(len(legs))
    rule3_contrib = np.zeros(len(legs))

    # Iterate through legs to compute score
    for k, leg in enumerate(legs):
        # For the first rule
        rule1_contrib[
            [leg_corresp_id[l] for l in rule1_corresponding_legs[leg]]
        ] += rule1_weight * (
            stepping_advancement[k] > 0
            and stepping_advancement[k] < leg_stance_starts[leg]
        ).astype(
            float
        )

        # For the second rule strong contact force happens at the beggining of the stance phase
        for l in rule2_corresponding_legs[leg]:
            # Decrease with stepping advancement
            if l[0] == leg[0]:
                # ipsilateral leg
                rule2_contrib[leg_corresp_id[l]] += (
                    rule2_weight
                    * (
                        (interp_step_duration - leg_stance_starts[leg])
                        - (stepping_advancement[k] - leg_stance_starts[leg])
                    )
                    if (stepping_advancement[k] - leg_stance_starts[leg]) > 0
                    else 0
                )
            else:
                # contralateral leg
                rule2_contrib[leg_corresp_id[l]] += (
                    rule2_weight_contralateral
                    * (
                        (interp_step_duration - leg_stance_starts[leg])
                        - (stepping_advancement[k] - leg_stance_starts[leg])
                    )
                    if (stepping_advancement[k] - leg_stance_starts[leg]) > 0
                    else 0
                )

        # For the third rule
        for l in rule3_corresponding_legs[leg]:
            # Increase with stepping advancement
            if l[0] == leg[0]:
                rule3_contrib[leg_corresp_id[l]] += (
                    rule3_weight * ((stepping_advancement[k] - leg_stance_starts[leg]))
                    if (stepping_advancement[k] - leg_stance_starts[leg]) > 0
                    else 0
                )
            else:
                rule3_contrib[leg_corresp_id[l]] += (
                    rule3_weight_contralateral
                    * ((stepping_advancement[k] - leg_stance_starts[leg]))
                    if (stepping_advancement[k] - leg_stance_starts[leg]) > 0
                    else 0
                )

    return rule1_contrib, rule2_contrib, rule3_contrib


######## Running variables ########

rule1_corresponding_legs = {
    "LH": ["LM"],
    "LM": ["LF"],
    "LF": [],
    "RH": ["RM"],
    "RM": ["RF"],
    "RF": [],
}
rule2_corresponding_legs = {
    "LH": ["LM", "RH"],
    "LM": ["LF", "RM"],
    "LF": ["RF"],
    "RH": ["RM", "LH"],
    "RM": ["RF", "LM"],
    "RF": ["LF"],
}
rule3_corresponding_legs = {
    "LH": ["RH"],
    "LM": ["LH", "RM"],
    "LF": ["LM", "RF"],
    "RH": ["LH"],
    "RM": ["RH", "LM"],
    "RF": ["LF", "RM"],
}

# Rule 1 should supress lift off (if a leg is in swing coupled legs should not be lifted most important leg to guarantee stability)
rule1_weight = -1e4
# Rule 2 should facilitate early protraction (upon touchdown of a leg coupled legs are encouraged to swing)
rule2_weight = 2.5
rule2_weight_contralateral = 1
# Rule 3 should enforce late protraction (the later in the stance the more it facilitates stance initiation)
rule3_weight = 3
rule3_weight_contralateral = 2

# one percent margin if leg score within this margin to the max score random choice between the very likely legs
percent_margin = 0.001
