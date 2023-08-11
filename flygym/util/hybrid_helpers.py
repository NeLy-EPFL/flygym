import numpy as np


def get_raise_leg(nmf):
    legs = [lts[:2] for lts in nmf.last_tarsalseg_names]
    joints_in_leg = [
        [i for i, joint in enumerate(nmf.actuated_joints) if leg in joint]
        for leg in legs
    ]
    joint_name_to_id = {
        joint[8:]: i
        for i, joint in enumerate(nmf.actuated_joints[: len(joints_in_leg[0])])
    }
    raise_leg = np.zeros((len(legs), 42))
    for i, leg in enumerate(legs):
        if "F" in leg:
            raise_leg[i, joints_in_leg[i][joint_name_to_id["Femur"]]] = -0.02
            raise_leg[i, joints_in_leg[i][joint_name_to_id["Tibia"]]] = +0.016
        elif "M" in leg:
            raise_leg[i, joints_in_leg[i][joint_name_to_id["Coxa"]]] = -0.015
            raise_leg[i, joints_in_leg[i][joint_name_to_id["Femur"]]] = 0.004
            raise_leg[i, joints_in_leg[i][joint_name_to_id["Tibia"]]] = 0.01
            raise_leg[i, joints_in_leg[i][joint_name_to_id["Tarsus1"]]] = -0.008
            raise_leg[i, joints_in_leg[i]] *= 1.2
        elif "H" in leg:
            raise_leg[i, joints_in_leg[i][joint_name_to_id["Femur"]]] = -0.01
            raise_leg[i, joints_in_leg[i][joint_name_to_id["Tibia"]]] = +0.005
    return raise_leg
