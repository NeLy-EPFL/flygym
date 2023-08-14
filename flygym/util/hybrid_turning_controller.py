import PIL.Image

import numpy as np
import pkg_resources
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo, MuJoCoParameters
from tqdm import trange
from flygym.util.config import all_leg_dofs
from flygym.state import stretched_pose

from cpg_controller import CPG

from gymnasium import spaces

import cv2


class NMFHybridTurning(NeuroMechFlyMuJoCo):
    def __init__(
        self,
        n_stabilisation_steps: int = 5000,
        **kwargs,
    ):
        # The underlying normal NMF environment
        super().__init__(**kwargs)
        # Number of dofs of the observation space
        self.num_dofs = len(self.actuated_joints)
        # Action space - 2 values (alphaL and alphaR)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        # CPG initialization
        self.cpg = CPG(self.timestep)
        self.n_stabilisation_steps = n_stabilisation_steps
        for _ in range(n_stabilisation_steps):
            self.cpg.step()

        # Processing of joint trajectories reference from stepping data
        self._load_preprogrammed_stepping()

        # Initialization of the rule-based part of the controller
        self._reset_leg_retraction_state()
        self._reset_stumble_state()
        self.enable_rules_delay = 500
        self.timer = 0

    def reset(self):
        self.cpg.reset()
        for _ in range(self.n_stabilisation_steps):
            self.cpg.step()
        self._reset_leg_retraction_state()
        self._reset_stumble_state()
        self.timer = 0
        return super().reset()

    def step(self, action):
        # Get observation before stepping
        obs = self.get_observation()

        # Scaling of the action to go from [-1,1] -> [-0.5,0.5]
        action = 0.5 * np.array(action)

        # Compute joint positions from NN output
        joints_action = self.compute_joints_cpg(action)
        if self.timer > self.enable_rules_delay:
            # Add action of rules
            joints_action += self.leg_retraction_rule(obs["end_effectors"][2::3]) 
            joints_action += self.stumble_rule()

        # Get adhesion signal
        adhesion_signal = np.zeros(6)

        self.timer +=1

        # Step simulation and get observation after it
        return super().step({
            "joints": joints_action, 
            "adhesion": adhesion_signal
        })

    def compute_joints_cpg(self, action):
        """Turn NN output into joint position."""
        self.cpg.step(action)
        indices = self._cpg_state_to_joint_state()

        joints_action = self.step_data[self.joint_ids, 0]
        joints_action += ( self.step_data[self.joint_ids, indices]
                        - self.step_data[self.joint_ids, 0] 
                        ) * self.cpg.amplitude[self.match_leg_to_joints]

        return joints_action
    
    def leg_retraction_rule(self, ee_z_pos):
        # Detect legs in hole, keep only the deepest
        legs_in_hole = ee_z_pos < self.floor_height
        legs_in_hole = np.logical_and(legs_in_hole, ee_z_pos == np.min(ee_z_pos))

        for k, tarsal_seg in enumerate(self.last_tarsalseg_names):
            if legs_in_hole[k]:
                self.legs_in_hole_increment[k] += self.increase_rate
            else:
                if self.legs_in_hole_increment[k] > 0:
                    self.legs_in_hole_increment[k] -= self.decrease_rate
                    
        return self.legs_in_hole_increment
    
    def stumble_rule(self):
        return 0

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


    def _cpg_state_to_joint_state(self):
        """From phase define what is the corresponding timepoint in the joint dataset
        In the case of the oscillator, the period is 2pi and the step duration is the period of the step
        We have to match those two"""
        period = 2 * np.pi
        # match length of step to period phases should have a period of period mathc this perios to the one of the step
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

        # Set floor height to lowest point in the walking parts of the floor
        floor_height = np.inf
        for i_g in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(i_g)
            name = geom.name
            if "groundblock" in name:
                block_height = geom.pos[2] + geom.size[2]
                floor_height = min(floor_height, block_height)
        floor_height -= 0.05  # account for small penetrations of the floor

        self.floor_height = floor_height
    
    def _reset_stumble_state(self):
        # Sensors to detect leg with "unatural" (other than tarsus 4 or 5) contacts
        self.leg_tarsus1T_contactsensors = [
            [
                i
                for i, cs in enumerate(self.contact_sensor_placements)
                if tarsal_seg[:2] in cs and ("Tibia" in cs or "Tarsus1" in cs)
            ]
            for tarsal_seg in self.last_tarsalseg_names
        ]

        # Parameters of the stimulus reaction
        self.force_threshold = 5.0
        self.increase_rate = 0.1
        self.decrease_rate = 0.05

        # Memorization variables
        self.highest_proximal_contact_leg = [False] * 6
        self.legs_w_proximalcontact_increment = np.zeros(6)

        # Conversion from contact force id to leg adhesion id
        self.last_tarsalseg_to_adh_id = [
            i
            for adh in self.adhesion_actuators
            for i, lts in enumerate(self.last_tarsalseg_names)
            if lts[:2] == adh.name[:2]
        ]