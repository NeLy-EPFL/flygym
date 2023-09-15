import numpy as np
import pkg_resources
import pickle
import logging
from gymnasium import spaces
from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters

from flygym.util.cpg_controller import CPG


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


class HybridTurningController(NeuroMechFlyMuJoCo):
    """A wrapper for the NMF class controller with a CPG-rule-based hybrid controller.

    Parameters
    ----------
    n_stabilisation_steps :
        Initial number of CPG steps to run before applying its output to the
        simulation, for stabilization.
    turn_mode :
        CPG parameters modulated by the action. Can be "amp", "freq" or "both".
    epsilon_turn :
        Minimum difference between the action terms to consider it a turning
        behaviour and disable corresponding leg adhesion.
    """

    def __init__(
        self,
        stabilisation_dur: int = 0.1,
        epsilon_turn: float = 0.1,
        **kwargs,
    ):
        if "sim_params" not in kwargs:
            kwargs["sim_params"] = MuJoCoParameters(enable_adhesion=True)
            logging.warning("Enabling adhesion for turning controller")
        if "spawn_pos" not in kwargs:
            kwargs["spawn_pos"] = [0, 0, 0.2]
            logging.warning(
                f"Setting spawn pos to {kwargs['sim_params']} for turning controller"
            )
        # The underlying normal NMF environment
        super().__init__(**kwargs)
        # Number of dofs of the observation space
        self.num_dofs = len(self.actuated_joints)

        self.n_stabilisation_steps = int(np.round(stabilisation_dur / self.timestep))

        # Action space - 2 values (alphaL and alphaR)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.eps_turn = epsilon_turn

        # CPG initialization
        self.cpg = CPG(timestep=self.timestep)
        for i in range(2000):
            self.cpg.step([0.0, 0.0])

        # Processing of joint trajectories reference from stepping data
        self._load_preprogrammed_stepping()

        # Initialization of the rule-based part of the controller
        self._init_raise_leg()
        self._reset_leg_retraction_state()
        self._reset_stumble_state()
        self.stumble_active = False
        self.leg_retract_active = False

        self.timer = 0

        for i in range(2000):
            self.cpg.step([0.0, 0.0])

    def run_stabilisation(self, render_onset=8000):
        phases = []
        amplitudes = []
        obs_list = []
        for i in range(self.n_stabilisation_steps):
            self.cpg.step([0.0, 0.0])
            joints_action = self.step_data[self.joint_ids, 0]
            adhesion_signal = np.ones(6)
            obs, reward, terminated, truncated, info = super().step(
                {"joints": joints_action, "adhesion": adhesion_signal}
            )
            if i > render_onset:
                self.render()
            phases.append(self.cpg.phase)
            amplitudes.append(self.cpg.amplitude)
            obs_list.append(obs)
        return phases, amplitudes, obs_list

    def reset(self):
        self.cpg.reset()
        self._reset_leg_retraction_state()
        self._reset_stumble_state()
        self.stumble_active = False
        self.leg_retract_active = False
        self.timer = 0
        return super().reset()

    def step(self, action):
        # Get observation before stepping
        obs = self.get_observation()

        # Compute joint positions from NN output
        joints_action = self.compute_joints_cpg(action)
        leg_in_stance = self.get_legs_in_stance()

        fly_z_pos = obs["fly"][0][2]
        ee_z_pos = obs["end_effectors"][2::3]
        leg_to_thorax_zdistance = fly_z_pos - ee_z_pos
        leg_retraction_contribution = self.increment_leg_retraction_rule(
            leg_to_thorax_zdistance
        )

        fly_orient = obs["fly_orient"]
        contact_forces = obs["contact_forces"]
        leg_stumble_contribution = self.increment_stumble_rule(
            fly_orient, contact_forces, leg_in_stance
        )

        joints_action += leg_retraction_contribution + leg_stumble_contribution

        # Get adhesion signal
        if self.sim_params.enable_adhesion:
            adhesion_signal = leg_in_stance
            rule_active_legs = np.logical_or(
                self.proximal_contact_leg, self.legs_in_hole
            )[self.last_tarsalseg_to_adh_id]
            adhesion_signal[rule_active_legs] = 0.0
        else:
            adhesion_signal = np.zeros(6)

        # Step simulation and get observation after it
        self.timer += 1
        return super().step({"joints": joints_action, "adhesion": adhesion_signal})

    def compute_joints_cpg(self, action):
        """Turn NN output into joint position."""
        self.cpg.step(action)
        indices = self._cpg_state_to_joint_state()

        joints_action = self.step_data[self.joint_ids, 0]

        joints_action += (
            self.step_data[self.joint_ids, indices] - self.step_data[self.joint_ids, 0]
        ) * self.cpg.amplitude[self.match_leg_to_joints]

        return joints_action

    def get_legs_in_stance(self):
        indices_in_order = self._cpg_state_to_joint_state()[self.joints_to_leg]
        leg_in_stance = np.logical_or(
            indices_in_order < self.swing_starts_in_order,
            indices_in_order > self.stance_starts_in_order,
        )

        return leg_in_stance

    def increment_leg_retraction_rule(self, leg_to_thorax_zdistance):
        active = 0
        # get the third furthest leg from the thorax (as tripod should be on the floor)
        third_furthest_leg = np.sort(leg_to_thorax_zdistance)[3]
        # print(np.sort(leg_to_thorax_zdistance))
        self.legs_in_hole = np.logical_and(
            leg_to_thorax_zdistance
            > third_furthest_leg + self.retraction_distance_margin,
            leg_to_thorax_zdistance == np.max(leg_to_thorax_zdistance),
        )
        for k, tarsal_seg in enumerate(self.last_tarsalseg_names):
            if self.legs_in_hole[k]:
                self.physics.named.model.geom_rgba[
                    "Animat/" + tarsal_seg[:2] + "Tibia_visual"
                ] = [0.0, 0.0, 1.0, 1.0]
                self.legs_in_hole_increment[k] += self.increase_rate_retraction
            else:
                self.physics.named.model.geom_rgba[
                    "Animat/" + tarsal_seg[:2] + "Tibia_visual"
                ] = self.base_rgba
                if self.legs_in_hole_increment[k] > 0:
                    self.legs_in_hole_increment[k] -= self.decrease_rate_retraction

        if active > 0:
            self.leg_retract_active = True
        else:
            self.leg_retract_active = False

        return (self.raise_leg.T * self.legs_in_hole_increment).sum(axis=1)

    def increment_stumble_rule(self, fly_orient, contact_forces, leg_in_stance):
        active = 0
        # Compute forces
        fly_orient[2] = 0.0
        fly_orient_norm = np.linalg.norm(fly_orient)
        fly_orient_unit = fly_orient / fly_orient_norm
        # Look for forces opposing the fly orientation (and thus progression)
        for k, contact_sensors in enumerate(self.leg_tarsus12T_contactsensors):
            sensor_contact_forces = contact_forces[:, contact_sensors].T
            min_scalar_proj_contact_force = np.inf
            for contact_force in sensor_contact_forces:
                scalar_proj_force = np.dot(
                    contact_force, fly_orient_unit
                )  # scalar_value
                min_scalar_proj_contact_force = min(
                    min_scalar_proj_contact_force, scalar_proj_force
                )
            self.proximal_contact_leg[k] = np.logical_and(
                min_scalar_proj_contact_force < self.stumbling_force_threshold,
                np.logical_not(leg_in_stance[k]),
            )
            self.projected_force[k] = min_scalar_proj_contact_force

        for k, tarsal_seg in enumerate(self.last_tarsalseg_names):
            if self.proximal_contact_leg[k] and not self.legs_in_hole[k]:
                self.physics.named.model.geom_rgba[
                    "Animat/" + tarsal_seg[:2] + "Femur_visual"
                ] = [1.0, 0.0, 0.0, 1.0]
                self.legs_w_proximalcontact_increment[k] += self.increase_rate_stumbling
            else:
                self.physics.named.model.geom_rgba[
                    "Animat/" + tarsal_seg[:2] + "Femur_visual"
                ] = self.base_rgba
                if self.legs_w_proximalcontact_increment[k] > 0:
                    self.legs_w_proximalcontact_increment[
                        k
                    ] -= self.decrease_rate_stumbling

        if active > 0:
            self.stumble_active = True
        else:
            self.stumble_active = False

        return (self.raise_leg.T * self.legs_w_proximalcontact_increment).sum(axis=1)

    def _load_preprogrammed_stepping(self):
        legs = ["LF", "LM", "LH", "RF", "RM", "RH"]
        n_joints = len(self.actuated_joints)
        self.joint_ids = np.arange(n_joints).astype(int)
        self.match_leg_to_joints = np.array(
            [
                i
                for joint in self.actuated_joints
                for i, leg in enumerate(legs)
                if leg in joint
            ]
        )

        # Load recorded data
        data_path = Path(pkg_resources.resource_filename("flygym", "data"))
        with open(data_path / "behavior" / "single_steps.pkl", "rb") as f:
            data = pickle.load(f)

        # Treatment of the pre-recorded data
        step_duration = len(data["joint_LFCoxa"])
        self.interp_step_duration = int(
            step_duration * data["meta"]["timestep"] / self.timestep
        )
        step_data_block_base = np.zeros(
            (len(self.actuated_joints), self.interp_step_duration)
        )
        measure_t = np.arange(step_duration) * data["meta"]["timestep"]
        interp_t = np.arange(self.interp_step_duration) * self.timestep
        for i, joint in enumerate(self.actuated_joints):
            step_data_block_base[i, :] = np.interp(interp_t, measure_t, data[joint])

        self.step_data = step_data_block_base.copy()

        leg_swing_starts = {
            k: round(v / self.timestep)
            for k, v in data["swing_stance_time"]["swing"].items()
        }
        leg_stance_starts = {
            k: round(v / self.timestep)
            for k, v in data["swing_stance_time"]["stance"].items()
        }

        self.joints_to_leg = np.array(
            [
                i
                for ts in self.last_tarsalseg_names
                for i, joint in enumerate(self.actuated_joints)
                if f"{ts[:2]}Coxa_roll" in joint
            ]
        )
        self.stance_starts_in_order = np.array(
            [leg_stance_starts[ts[:2]] for ts in self.last_tarsalseg_names]
        )
        self.swing_starts_in_order = np.array(
            [leg_swing_starts[ts[:2]] for ts in self.last_tarsalseg_names]
        )

    def _cpg_state_to_joint_state(self):
        """From phase define what is the corresponding timepoint in the joint dataset
        In the case of the oscillator, the period is 2pi and the step duration is the period of the step
        We have to match those two"""
        period = 2 * np.pi
        # match length of step to period phases should have a period of period match this period to the one of the step
        t_indices = np.round(
            np.mod(
                self.cpg.phase * self.interp_step_duration / period,
                self.interp_step_duration - 1,
            )
        ).astype(int)
        t_indices = t_indices[self.match_leg_to_joints]
        return t_indices

    def _reset_leg_retraction_state(self):
        self.legs_in_hole = [False] * 6
        self.legs_in_hole_increment = np.zeros(6)
        self.retraction_distance_margin = 0.05
        self.increase_rate_retraction = 1 / 2e-3 * self.timestep
        self.decrease_rate_retraction = 1 / 3e-3 * self.timestep

    def _reset_stumble_state(self):
        # Sensors to detect leg with "unatural" (other than tarsus 4 or 5) contacts
        self.leg_tarsus12T_contactsensors = [
            [
                i
                for i, cs in enumerate(self.contact_sensor_placements)
                if tarsal_seg[:2] in cs
                and ("Tibia" in cs or "Tarsus1" in cs or "Tarsus2" in cs)
            ]
            for tarsal_seg in self.last_tarsalseg_names
        ]
        self.stumbling_force_threshold = -1.0
        self.increase_rate_stumbling = 1 / 5e-4 * self.timestep  # 1 step every 500µs
        self.decrease_rate_stumbling = 1 / 2e-3 * self.timestep

        # Memorization variables
        self.proximal_contact_leg = [False] * 6
        self.legs_w_proximalcontact_increment = np.zeros(6)
        self.projected_force = np.zeros(6)

        # Conversion from contact force id to leg adhesion id
        self.last_tarsalseg_to_adh_id = [
            i
            for adh in self.adhesion_actuators
            for i, lts in enumerate(self.last_tarsalseg_names)
            if lts[:2] == adh.name[:2]
        ]

    def _init_raise_leg(self):
        legs = [lts[:2] for lts in self.last_tarsalseg_names]
        joints_in_leg = [
            [i for i, joint in enumerate(self.actuated_joints) if leg in joint]
            for leg in legs
        ]
        joint_name_to_id = {
            joint[8:]: i
            for i, joint in enumerate(self.actuated_joints[: len(joints_in_leg[0])])
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

        self.raise_leg = raise_leg
