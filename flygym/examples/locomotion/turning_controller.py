import numpy as np
import warnings
from tqdm import trange
from scipy.interpolate import interp1d
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from dm_control.rl.control import PhysicsError

from flygym.fly import Fly
from flygym.simulation import SingleFlySimulation
from flygym.preprogrammed import all_leg_dofs
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
from flygym.arena import MixedTerrain


_tripod_phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
_tripod_coupling_weights = (_tripod_phase_biases > 0) * 10

_default_correction_vectors = {
    # "leg pos": (Coxa, Coxa_roll, Coxa_yaw, Femur, Femur_roll, Tibia, Tarsus1)
    # unit: radian
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}

_default_correction_rates = {"retraction": (800, 700), "stumbling": (2200, 1800)}


class HybridTurningController(SingleFlySimulation):
    """
    This class implements a controller that uses a CPG network to generate
    leg movements and uses a set of sensory-based rules to correct for
    stumbling and retraction. The controller also receives a 2D descending
    input to modulate the amplitudes and frequencies of the CPGs to
    accomplish turning.

    Notes
    -----
    Please refer to the `"MPD Task Specifications" page
    <https://neuromechfly.org/api_ref/mdp_specs.html#hybrid-turning-controller-hybridturningcontroller>`_
    of the API references for the detailed specifications of the action
    space, the observation space, the reward, the "terminated" and
    "truncated" flags, and the "info" dictionary.
    """

    def __init__(
        self,
        fly: Fly,
        preprogrammed_steps=None,
        intrinsic_freqs=np.ones(6) * 12,
        intrinsic_amps=np.ones(6) * 1,
        phase_biases=_tripod_phase_biases,
        coupling_weights=_tripod_coupling_weights,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        stumble_segments=("Tibia", "Tarsus1", "Tarsus2"),
        stumbling_force_threshold=-1,
        correction_vectors=_default_correction_vectors,
        correction_rates=_default_correction_rates,
        amplitude_range=(-0.5, 1.5),
        draw_corrections=False,
        max_increment=80 / 1e-4,
        retraction_persistence_duration=20 / 1e-4,
        retraction_persistence_initiation_threshold=20 / 1e-4,
        seed=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fly : Fly
            The fly object to be simulated.
        preprogrammed_steps : PreprogrammedSteps, optional
            Preprogrammed steps to be used for leg movement.
        intrinsic_freqs : np.ndarray, optional
            Intrinsic frequencies of the CPGs. See ``CPGNetwork`` for
            details.
        intrinsic_amps : np.ndarray, optional
            Intrinsic amplitudes of the CPGs. See ``CPGNetwork`` for
            details.
        phase_biases : np.ndarray, optional
            Phase biases of the CPGs. See ``CPGNetwork`` for details.
        coupling_weights : np.ndarray, optional
            Coupling weights of the CPGs. See ``CPGNetwork`` for details.
        convergence_coefs : np.ndarray, optional
            Convergence coefficients of the CPGs. See ``CPGNetwork`` for
            details.
        init_phases : np.ndarray, optional
            Initial phases of the CPGs. See ``CPGNetwork`` for details.
        init_magnitudes : np.ndarray, optional
            Initial magnitudes of the CPGs. See ``CPGNetwork`` for details.
        stumble_segments : tuple, optional
            Leg segments to be used for stumbling detection.
        stumbling_force_threshold : float, optional
            Threshold for stumbling detection.
        correction_vectors : dict, optional
            Correction vectors for each leg.
        correction_rates : dict, optional
            Correction rates for retraction and stumbling.
        amplitude_range : tuple, optional
            Range for leg lifting correction.
        draw_corrections : bool, optional
            Whether to color-code legs to indicate if correction rules
            are active in the rendered video.
        max_increment : float, optional
            Maximum duration of the correction before it is capped.
        retraction_persist3nce_duration : float, optional
            Time spend in a persistent state (leg is further retracted)
            even if the rule is no longer active
        retraction_persist3nce_initiation_threshold : float, optional
            Amount of time the leg had to be retracted for for the persistence
            to be initiated (prevents activation of persistence for noise driven
            rule activations)
        seed : int, optional
            Seed for the random number generator.
        **kwargs
            Additional keyword arguments to be passed to
            ``SingleFlySimulation.__init__``.
        """
        # Check if we have the correct list of actuated joints
        if fly.actuated_joints != all_leg_dofs:
            raise ValueError(
                "``HybridTurningController`` requires a specific set of DoFs, namely "
                "``flygym.preprogrammed.all_leg_dofs``, to be actuated. A different "
                "set of DoFs was provided."
            )

        # Initialize core NMF simulation
        super().__init__(fly=fly, **kwargs)

        if preprogrammed_steps is None:
            preprogrammed_steps = PreprogrammedSteps()
        self.preprogrammed_steps = preprogrammed_steps
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.phase_biases = phase_biases
        self.coupling_weights = coupling_weights
        self.convergence_coefs = convergence_coefs
        self.stumble_segments = stumble_segments
        self.stumbling_force_threshold = stumbling_force_threshold
        self.correction_vectors = correction_vectors
        self.correction_rates = correction_rates
        self.amplitude_range = amplitude_range
        self.draw_corrections = draw_corrections
        self.max_increment = max_increment * self.timestep
        self.retraction_persistence_duration = (
            retraction_persistence_duration * self.timestep
        )
        self.retraction_persistence_initiation_threshold = (
            retraction_persistence_initiation_threshold * self.timestep
        )
        self.retraction_persistence_counter = np.zeros(6)
        # Define the joints that need to be inverted to
        # mirror actions from left to right
        self.right_leg_inversion = [1, -1, -1, 1, -1, 1, 1]

        # Define action and observation spaces
        self.action_space = spaces.Box(*amplitude_range, shape=(2,))

        # Initialize CPG network
        self.cpg_network = CPGNetwork(
            timestep=self.timestep,
            intrinsic_freqs=intrinsic_freqs,
            intrinsic_amps=intrinsic_amps,
            coupling_weights=coupling_weights,
            phase_biases=phase_biases,
            convergence_coefs=convergence_coefs,
            init_phases=init_phases,
            init_magnitudes=init_magnitudes,
            seed=seed,
        )

        # Initialize variables tracking the correction amount
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)

        # Find stumbling sensors
        self.stumbling_sensors = self._find_stumbling_sensor_indices()
        # Define the gain applied to the correction based on the phase of the CPG
        # (retracting the leg more during the swing phase, less during the stance phase)
        self.phasic_multiplier = self._init_phasic_gain()

    def _init_phasic_gain(self, swing_extension=np.pi / 4):
        """
        Initialize the gain applied to the correction based on the phase of
        the CPG. Lengthen the swing phase by swing_extension to give more
        chances to the leg to avoid obstacles.
        """

        phasic_multiplier = {}

        for leg in self.preprogrammed_steps.legs:
            swing_start, swing_end = self.preprogrammed_steps.swing_period[leg]

            step_points = [
                swing_start,
                np.mean([swing_start, swing_end]),
                swing_end + swing_extension,
                np.mean([swing_end, 2 * np.pi]),
                2 * np.pi,
            ]
            self.preprogrammed_steps.swing_period[leg] = (
                swing_start,
                swing_end + swing_extension,
            )
            increment_vals = [0, 0.8, 0, -0.1, 0]

            phasic_multiplier[leg] = interp1d(
                step_points, increment_vals, kind="linear", fill_value="extrapolate"
            )

        return phasic_multiplier

    def _find_stumbling_sensor_indices(self):
        """Find the indices of the sensors that are used for stumbling detection."""
        stumbling_sensors = {leg: [] for leg in self.preprogrammed_steps.legs}
        for i, sensor_name in enumerate(self.fly.contact_sensor_placements):
            leg = sensor_name.split("/")[1][:2]  # sensor_name: e.g. "Animat/LFTarsus1"
            segment = sensor_name.split("/")[1][2:]
            if segment in self.stumble_segments:
                stumbling_sensors[leg].append(i)
        stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}
        if any(
            v.size != len(self.stumble_segments) for v in stumbling_sensors.values()
        ):
            raise RuntimeError(
                "Contact detection must be enabled for all tibia, tarsus1, and tarsus2 "
                "segments for stumbling detection."
            )
        return stumbling_sensors

    def _retraction_rule_find_leg(self, obs):
        """
        Returns the index of the leg that needs to be retracted, or None
        if none applies.
        Retraction can be due to the activation of a rule or persistence.
        Every time the rule is active the persistence counter is set to 1.
        At every step the persistence counter is incremented. If the rule
        is still active it is again reset to 1 otherwise, it will be
        incremented until it reaches the persistence duration. At this
        point the persistence counter is reset to 0.
        """
        end_effector_z_pos = obs["fly"][0][2] - obs["end_effectors"][:, 2]
        end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
        end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
        if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
            leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
            if (
                self.retraction_correction[leg_to_correct_retraction]
                > self.retraction_persistence_initiation_threshold
            ):
                self.retraction_persistence_counter[leg_to_correct_retraction] = 1
        else:
            leg_to_correct_retraction = None
        return leg_to_correct_retraction

    def _update_persistence_counter(self):
        """
        Increment the persistence counter if it is nonzero. Zero the
        counter when it reaches the persistence duration."""
        # increment every nonzero counter
        self.retraction_persistence_counter[
            self.retraction_persistence_counter > 0
        ] += 1
        # zero the increment when reaching the threshold
        self.retraction_persistence_counter[
            self.retraction_persistence_counter > self.retraction_persistence_duration
        ] = 0

    def _stumbling_rule_check_condition(self, obs, leg):
        """Return True if the leg is stumbling, False otherwise."""
        # update stumbling correction amounts
        contact_forces = obs["contact_forces"][self.stumbling_sensors[leg], :]
        fly_orientation = obs["fly_orientation"]
        # force projection should be negative if against fly orientation
        force_proj = np.dot(contact_forces, fly_orientation)
        return (force_proj < self.stumbling_force_threshold).any()

    def _get_net_correction(self, retraction_correction, stumbling_correction):
        """Retraction correction has priority."""
        if retraction_correction > 0:
            return retraction_correction, True
        return stumbling_correction, False

    def _update_correction_amount(
        self, condition, curr_amount, correction_rates, viz_segment
    ):
        """Update correction amount and color code leg segment.

        Parameters
        ----------
        condition : bool
            Whether the correction condition is met.
        curr_amount : float
            Current correction amount.
        correction_rates : tuple[float, float]
            Correction rates for increment and decrement.
        viz_segment : str
            Name of the segment to color code. If None, no color coding is
            done.

        Returns
        -------
        float
            Updated correction amount.
        """
        if condition:  # lift leg
            increment = correction_rates[0] * self.timestep
            new_amount = curr_amount + increment
            color = (1, 0, 0, 1)
        else:  # condition no longer met, lower leg
            decrement = correction_rates[1] * self.timestep
            new_amount = max(0, curr_amount - decrement)
            color = (0.5, 0.5, 0.5, 1)
        if viz_segment is not None:
            self.fly.change_segment_color(self.physics, viz_segment, color)
        return new_amount, condition

    def reset(self, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        """
        Reset the simulation.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. If None, the simulation
            is re-seeded without a specific seed. For reproducibility,
            always specify a seed.
        init_phases : np.ndarray, optional
            Initial phases of the CPGs. See ``CPGNetwork`` for details.
        init_magnitudes : np.ndarray, optional
            Initial magnitudes of the CPGs. See ``CPGNetwork`` for details.
        **kwargs
            Additional keyword arguments to be passed to
            ``SingleFlySimulation.reset``.

        Returns
        -------
        np.ndarray
            Initial observation upon reset.
        dict
            Additional information.
        """
        obs, info = super().reset(seed=seed)
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)
        return obs, info

    def step(self, action):
        """Step the simulation forward one timestep.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (2,) containing descending signal encoding
            turning.
        """
        # update CPG parameters
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        # get current observation
        obs = super().get_observation()

        # Retraction rule: is any leg stuck in a gap and needing to be retracted?
        leg_to_correct_retraction = self._retraction_rule_find_leg(obs)
        self._update_persistence_counter()
        persistent_retraction = self.retraction_persistence_counter > 0

        self.cpg_network.step()

        joints_angles = []
        adhesion_onoff = []
        all_net_corrections = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # update retraction correction amounts
            retraction_correction, is_retracted = self._update_correction_amount(
                condition=(
                    (i == leg_to_correct_retraction) or persistent_retraction[i]
                ),  # lift leg
                curr_amount=self.retraction_correction[i],
                correction_rates=self.correction_rates["retraction"],
                viz_segment=f"{leg}Tibia" if self.draw_corrections else None,
            )
            self.retraction_correction[i] = retraction_correction

            # update stumbling correction amounts
            self.stumbling_correction[i], is_stumbling = self._update_correction_amount(
                condition=self._stumbling_rule_check_condition(obs, leg),
                curr_amount=self.stumbling_correction[i],
                correction_rates=self.correction_rates["stumbling"],
                viz_segment=(
                    f"{leg}Femur"
                    if self.draw_corrections and retraction_correction <= 0
                    else None
                ),
            )

            # get net correction amount
            net_correction, reset_stumbling = self._get_net_correction(
                self.retraction_correction[i], self.stumbling_correction[i]
            )
            if reset_stumbling:
                self.stumbling_correction[i] = 0.0

            net_correction = np.clip(net_correction, 0, self.max_increment)
            if leg[0] == "R":
                net_correction *= self.right_leg_inversion[i]

            net_correction *= self.phasic_multiplier[leg](
                self.cpg_network.curr_phases[i] % (2 * np.pi)
            )

            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            my_joints_angles += net_correction * self.correction_vectors[leg[1]]
            joints_angles.append(my_joints_angles)

            all_net_corrections.append(net_correction)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )

            # No adhesion in stumbling or retracted
            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }
        obs, reward, terminated, truncated, info = super().step(action)
        info["net_corrections"] = np.array(all_net_corrections)
        info.update(action)  # add lower-level action to info
        return obs, reward, terminated, truncated, info


class HybridTurningNMF(HybridTurningController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            (
                "`HybridTurningNMF` has been renamed `HybridTurningController` ."
                "Please use `HybridTurningController`. `HybridTurningNMF` is "
                "deprecated and will be removed in a future release."
            ),
            DeprecationWarning,
        )


if __name__ == "__main__":
    from flygym import Fly, Camera

    run_time = 1.0
    timestep = 1e-4
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    np.random.seed(0)

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=contact_sensor_placements,
    )

    cam = Camera(fly=fly, camera_id="Animat/camera_right", play_speed=0.1)
    sim = HybridTurningController(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        seed=0,
        draw_corrections=True,
        arena=MixedTerrain(),
    )
    check_env(sim)

    obs_list = []

    obs, info = sim.reset(0)
    print(f"Spawning fly at {obs['fly'][0]} mm")

    for i in trange(int(run_time / sim.timestep)):
        curr_time = i * sim.timestep

        # To demonstrate left and right turns:
        if curr_time < run_time / 2:
            action = np.array([1.2, 0.4])
        else:
            action = np.array([0.4, 1.2])

        # To demonstrate that the result is identical with the hybrid controller without
        # turning:
        action = np.array([1.0, 1.0])

        try:
            obs, reward, terminated, truncated, info = sim.step(action)
            obs_list.append(obs)
            sim.render()
        except PhysicsError:
            print("Simulation was interrupted because of a physics error")
            break

    x_pos = obs_list[-1]["fly"][0][0]
    print(f"Final x position: {x_pos:.4f} mm")
    print(f"Simulation terminated: {obs_list[-1]['fly'][0] - obs_list[0]['fly'][0]}")

    cam.save_video("./outputs/hybrid_turning.mp4", 0)
