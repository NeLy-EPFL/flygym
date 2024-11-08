import numpy as np
from gymnasium import spaces

from flygym.fly import Fly
from flygym.simulation import SingleFlySimulation
from flygym.preprogrammed import all_leg_dofs
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork

from flygym.arena import BaseArena
from scipy.spatial.transform import Rotation as R

from gymnasium import spaces
from gymnasium.core import ObsType

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

class TurningController(SingleFlySimulation):
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
    seed : int, optional
        Seed for the random number generator.
    amplitude_range: tuple, optional
        Range of descending signals that can be applied to the CPGs.
    init_control_mode : str, optional
        Initial control mode. Can be "CPG", "Single", or "Tripod".
    leg_step_time : float, optional
        Time taken to step a leg in seconds.
    **kwargs
        Additional keyword arguments to be passed to
        ``SingleFlySimulation.__init__``.
    """

    def __init__(
        self,
        fly: Fly,
        preprogrammed_steps=None,
        intrinsic_freqs=np.ones(6) * 36, #np.ones(6) * 12,
        intrinsic_amps=np.ones(6) * 6,
        phase_biases=_tripod_phase_biases,
        coupling_weights=_tripod_coupling_weights,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        amplitude_range=(-0.5, 1.5),
        seed=0,
        init_control_mode="CPG",
        leg_step_time=0.025,
        **kwargs,
    ):
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

        self.prev_control_mode = init_control_mode

        self.leg_phases = np.zeros(6)
        self.phase_increment = self.timestep / leg_step_time * 2 * np.pi

        self.tripod_map = {"LF": 0 , "LM": 1, "LH":0, "RF": 1, "RM": 0, "RH": 1}
        self.tripod_phases = np.zeros(2)

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
        return obs, info
    
    def swap_control_mode(self):
        self.tripod_phases = np.zeros(2)
        self.leg_phases = np.zeros(6)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
    
    def get_cpg_joint_angles(self, action):
        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        self.cpg_network.step()

        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )

            # No adhesion in stumbling or retracted
            adhesion_onoff.append(my_adhesion_onoff)

        return {"joints": np.array(np.concatenate(joints_angles)),
                "adhesion": np.array(adhesion_onoff).astype(int)}


    def get_single_leg_joint_angles(self, action):
        # check if new legs need to be stepped
        
        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            if self.leg_phases[i] >= 2 * np.pi:
                self.leg_phases[i] = 0
            elif self.leg_phases[i] <= 0:
                if action[i] > 0:
                    self.leg_phases[i] += self.phase_increment
            else:
                self.leg_phases[i] += self.phase_increment

            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.leg_phases[i],
                1,
            )
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.leg_phases[i]
            )
            adhesion_onoff.append(my_adhesion_onoff)


        return {"joints": np.array(np.concatenate(joints_angles)), 
                "adhesion": np.array(adhesion_onoff).astype(int)}                
        

    def get_tripod_joint_angles(self, action):

        joints_angles = []
        adhesion_onoff = []

        for i in range(2):
            if self.tripod_phases[i] >= 2 * np.pi:
                self.tripod_phases[i] = 0
            elif self.tripod_phases[i] <= 0:
                if action[i] > 0:
                    self.tripod_phases[i] += self.phase_increment
            else:
                self.tripod_phases[i] += self.phase_increment
        
        for leg in self.preprogrammed_steps.legs:
            tripod_idx = self.tripod_map[leg]
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.tripod_phases[tripod_idx],
                1,
            )
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.tripod_phases[tripod_idx]
            )
            adhesion_onoff.append(my_adhesion_onoff)
        
        return {"joints": np.array(np.concatenate(joints_angles)), 
                "adhesion": np.array(adhesion_onoff).astype(int)}


    def step(self, action, control_mode):
        """Step the simulation forward one timestep.

        Parameters
        ----------
        action : np.ndarray
            Array of shape (2,) containing descending signal encoding
            turning.
        activated_legs : np.ndarray
            Array of shape (6,) containing the legs we want to activate
        """

        if control_mode != self.prev_control_mode:
            self.swap_control_mode()
        self.prev_control_mode = control_mode


        if control_mode == "CPG":
            joints_action = self.get_cpg_joint_angles(action)
        elif control_mode == "single":
            joints_action = self.get_single_leg_joint_angles(action)
        elif control_mode == "tripod":
            joints_action = self.get_tripod_joint_angles(action)
        else:
            raise ValueError("Invalid control mode")

        obs, reward, terminated, truncated, info = super().step(joints_action)
        info.update(joints_action)  # add lower-level action to info
        return obs, reward, terminated, truncated, info


class GameFly(Fly):
 def get_observation(self, sim: "Simulation") -> ObsType:
        """Get observation without stepping the physics simulation.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        """
        physics = sim.physics

        # fly position and orientation
        cart_pos = physics.bind(self._body_sensors[0]).sensordata
        cart_vel = physics.bind(self._body_sensors[1]).sensordata

        quat = physics.bind(self._body_sensors[2]).sensordata
        # ang_pos = transformations.quat_to_euler(quat)
        ang_pos = R.from_quat(quat[[1, 2, 3, 0]]).as_euler(
            "ZYX"
        )  # explicitly use extrinsic ZYX
        # ang_pos[0] *= -1  # flip roll??
        ang_vel = physics.bind(self._body_sensors[3]).sensordata
        fly_pos = np.array([cart_pos, cart_vel, ang_pos, ang_vel])

        self.last_obs["rot"] = ang_pos
        self.last_obs["pos"] = cart_pos

        orientation_vec = physics.bind(self._body_sensors[4]).sensordata.copy()

        obs = {
            "fly": fly_pos.astype(np.float32),
            "fly_orientation": orientation_vec.astype(np.float32),
        }

        # olfaction
        if self.enable_olfaction:
            antennae_pos = physics.bind(self._antennae_sensors).sensordata
            odor_intensity = sim.arena.get_olfaction(antennae_pos.reshape(4, 3))
            obs["odor_intensity"] = odor_intensity.astype(np.float32)

        # vision
        if self.enable_vision:
            self._update_vision(sim)
            obs["vision"] = self._curr_visual_input.astype(np.float32)

        return obs
 
 def _define_observation_space(self, arena: BaseArena):
        _observation_space = {
            "fly": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3)),
            "fly_orientation": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        }
        if self.enable_vision:
            _observation_space["vision"] = spaces.Box(
                low=0,
                high=255,
                shape=(2, self.config["vision"]["num_ommatidia_per_eye"], 2),
            )
        if self.enable_olfaction:
            _observation_space["odor_intensity"] = spaces.Box(
                low=0,
                high=np.inf,
                shape=(arena.odor_dimensions, len(self._antennae_sensors)),
            )
        return spaces.Dict(_observation_space)