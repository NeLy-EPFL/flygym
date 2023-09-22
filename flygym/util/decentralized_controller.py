import numpy as np

######## Initialisation ########

######## Running variables ########


def update_stepping_advancement(
    stepping_advancement, legs, interp_step_duration, increment=1
):
    # Advance the stepping advancement of each leg that are stepping, reset the advancement of the legs that are done stepping
    for k, leg in enumerate(legs):
        if stepping_advancement[k] >= interp_step_duration - 1:
            stepping_advancement[k] = 0
        elif stepping_advancement[k] > 0:
            stepping_advancement[k] += increment
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
