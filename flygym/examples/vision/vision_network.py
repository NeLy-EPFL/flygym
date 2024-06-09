import numpy as np
import torch
import flyvision
import warnings
from torch import Tensor
from typing import Union, Optional
from flyvision.utils.tensor_utils import AutoDeref
from flyvision.network import Network, NetworkView, IntegrationWarning
from flyvision.rendering import BoxEye
from flygym.vision import Retina


device = torch.device("cpu")
torch.set_default_device(device)
flyvision.device = torch.device(device)


class RealTimeVisionNetwork(Network):
    """
    This class extends ``flyvision.network.Network``. The main difference
    is that ``flyvision.network.Network`` receives the entire history of
    visual input as a block, which enables more efficient computation on
    the GPU. In contrast, ``RealTimeVisionNetwork`` receives visual input
    one frame at a time, allowing for deployment in closed-loop
    simulations. See `flyvision`_ and `Lappalainen et al., 2023`_ for
    more details.

    .. _flyvision: https://github.com/TuragaLab/flyvis
    .. _Lappalainen et al., 2023: https://www.biorxiv.org/content/10.1101/2023.03.11.532232
    """

    def setup_step_by_step_simulation(
        self,
        dt: float,
        initial_state: Union[str, AutoDeref, None] = "auto",
        as_states: bool = False,
        num_samples: int = 1,
    ) -> None:
        """
        Set up the network for step-by-step simulation.

        Parameters
        ----------
        dt : float
            Integration time step for the visual system neural network
            simulation. Note that this is typically different from (larger
            than) the time step of the physics simulation.
        initial_state : Union[str, AutoDeref, None], optional
            Initial state of the network. The default is "auto", which
            establishes a steady state after 1s of gray input. See
            ``RealTimeVisionNetwork.steady_state`` for more details.
        as_states : bool, optional
            Whether to return the network state or just the activities of
            the nodes. The default is False.
        num_samples : int, optional
            Number of samples to simulate in parallel. The default is 1.
            The user might want to change this to 2 since there are two
            compound eyes in the fly.
        """

        # Check dt
        if dt > 1 / 50:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    (
                        f"dt={dt} is very large for integration."
                        " better choose a smaller dt (<= 1/50 to avoid this warning)"
                    ),
                    IntegrationWarning,
                    stacklevel=2,
                )

        # Remember training-related states and set them to not-training
        is_training = self.training
        self.training = False
        params_requiring_grad = {}
        for name, params in self.named_parameters():
            params_requiring_grad[name] = params.requires_grad
            params.requires_grad = False

        # Keep parameters within their valid domain
        self._clamp()

        # Construct the parameter API
        params = self._param_api()

        # Set up initial state
        if initial_state is None:
            initial_state = self._initial_state(params=params, batch_size=1)
        elif initial_state == "auto":
            initial_state = self.steady_state(t_pre=1.0, dt=dt, batch_size=1)
        self._current_step_by_step_sim_state = initial_state

        # Identify input cells
        layer_index = {
            cell_type: index[:]
            for cell_type, index in self.connectome.nodes.layer_index.items()
        }
        input_node_index = np.array(
            [
                layer_index[cell_type.decode()]
                for cell_type in self.connectome.input_cell_types[:]
            ]
        )

        # Save parameters that will be used throughout the simulation or at cleanup
        self._step_by_step_sim_params = {
            "dt": dt,
            "is_training": is_training,
            "params_requiring_grad": params_requiring_grad,
            "as_states": as_states,
            "sim_params": params,
            "num_nodes": len(self.connectome.nodes.type),
            "num_samples": num_samples,
            "input_node_index": input_node_index,
        }

    def cleanup_step_by_step_simulation(self) -> None:
        """
        Clean up the network after the simulation ends. This clears the
        parameters that were set up for the step-by-step simulation and
        resets the gradient tracking / training flags.
        """
        self._check_step_by_step_simulation_setup()
        self.training = self._step_by_step_sim_params["is_training"]
        for name, params in self.named_parameters():
            params.requires_grad = self._step_by_step_sim_params[
                "params_requiring_grad"
            ][name]
        del self._step_by_step_sim_params
        del self._current_step_by_step_sim_state

    def forward_one_step(self, curr_visual_input: Tensor) -> Union[AutoDeref, Tensor]:
        """Simulate the network one step forward.

        Parameters
        ----------
        curr_visual_input : Tensor
            Raw visual input experienced by the fly (i.e., intensity
            reading from each ommatidium). This is a tensor of shape
            (num_samples, num_ommatidia).

        Returns
        -------
        Union[AutoDeref, Tensor]
            If ``as_states`` is set to True in ``__init__``, this returns
            the network state after stepping the network simulation by one
            step. The return value is of type
            ``flyvision.utils.tensor_utils.AutoDeref``. Otherwise, the
            actual activities of the nodes are returned as a torch tensor
            (i.e., agnostic to the flyvision state representation).
        """
        self._check_step_by_step_simulation_setup()

        # Convert visual input to stimulus state for all cells
        stimulus_buffer = torch.zeros(
            (
                self._step_by_step_sim_params["num_samples"],
                self._step_by_step_sim_params["num_nodes"],
            )
        )
        input_node_index = self._step_by_step_sim_params["input_node_index"]
        curr_visual_input = curr_visual_input.to(stimulus_buffer.device)
        for i in range(self._step_by_step_sim_params["num_samples"]):
            stimulus_buffer[i, input_node_index] += curr_visual_input[i]

        # Simulate one step
        self._current_step_by_step_sim_state = self._next_state(
            params=self._step_by_step_sim_params["sim_params"],
            state=self._current_step_by_step_sim_state,
            x_t=stimulus_buffer,
            dt=self._step_by_step_sim_params["dt"],
        )
        if self._step_by_step_sim_params["as_states"]:
            return self._current_step_by_step_sim_state
        else:
            return self._current_step_by_step_sim_state.nodes.activity

    def _check_step_by_step_simulation_setup(self) -> None:
        if not hasattr(self, "_step_by_step_sim_params"):
            raise RuntimeError(
                "You must call ``.setup_step_by_step_simulation(...)`` before calling "
                "``.forward_one_step(...)`` or ``.cleanup_step_by_step_simulation()``"
            )


class RealTimeVisionNetworkView(NetworkView):
    """
    This class extends ``flyvision.network.NetworkView`` to work with our
    extended ``RealTimeVisionNetwork``. In brief, it is used as a handle to
    set up the ``RealTimeVisionNetwork`` from saved checkpoint. See
    `flyvision`_ and `Lappalainen et al., 2023`_ for more details.

    .. _flyvision: https://github.com/TuragaLab/flyvis
    .. _Lappalainen et al., 2023: https://www.biorxiv.org/content/10.1101/2023.03.11.532232
    """

    def init_network(
        self, chkpt="best_chkpt", network: Optional[RealTimeVisionNetwork] = None
    ) -> Network:
        """
        Initialize the pretrained network.

        Parameters
        ----------
        chkpt: str
            Checkpoint to load. Default: "best_chkpt".
        network: RealTimeVisionNetwork, optional
            Network instance to initialize. If None, a new instance will
            be created.

        Returns
        -------
        RealTimeVisionNetwork
            The network instance.
        """
        if self._initialized["network"] and network is None:
            return self.network
        self.network = network or RealTimeVisionNetwork(**self.dir.config.network)
        state_dict = torch.load(self.dir / chkpt, map_location=flyvision.device)
        self.network.load_state_dict(state_dict["network"])
        self._initialized["network"] = True
        return self.network


class RetinaMapper:
    """
    Both flyvision and flygym use a hexagonal grid of ommatidia to model
    the compound eyes of the fly. To approximate the the correct number of
    ommatidia per eye (about 700-800), the two libraries even share the
    same size of the grid. However, the two libraries use different
    indexing conventions for the ommatidia. This class provides methods to
    convert stimuli between the coordinate systems of flyvision's
    ``BoxEye`` representation and flygym's ``Retina`` representation.
    """

    def __init__(
        self, retina: Optional[Retina] = None, boxeye: Optional[BoxEye] = None
    ):
        if retina is None:
            retina = Retina()
        if boxeye is None:
            boxeye = BoxEye(extent=15)

        # Let's call the center of the top ommatidium on the left-most row *a*
        # and ........................ bottom .............. right-most .. *b*

        # Get the coords of *a* and *b* on the BoxEye.receptor_centers
        receptor_centers = boxeye.receptor_centers.cpu().numpy()
        flyvis_centers_horiz_grid = sorted(np.unique(receptor_centers[:, 1]))
        col_min = flyvis_centers_horiz_grid[0]
        col_max = flyvis_centers_horiz_grid[-1]
        coords_leftmost_col = receptor_centers[receptor_centers[:, 1] == col_min]
        coords_rightmost_col = receptor_centers[receptor_centers[:, 1] == col_max]
        flyvis_a_coords = np.array([coords_leftmost_col[:, 0].min(), col_min])
        flyvis_b_coords = np.array([coords_rightmost_col[:, 0].max(), col_max])

        # Now, do the same for flygym.mujoco.vision.Retina
        retina = Retina()
        is_cell_a = retina.ommatidia_id_map == 1
        a_rows, a_cols = np.where(is_cell_a)
        flygym_a_coords = np.array([a_rows.mean(), a_cols.mean()])
        is_cell_b = retina.ommatidia_id_map == retina.ommatidia_id_map.max()
        b_rows, b_cols = np.where(is_cell_b)
        flygym_b_coord = np.array([b_rows.mean(), b_cols.mean()])

        # Establish linear mapping in row & col coordinates FROM flyvis TO flygym
        row_k = (flygym_b_coord[0] - flygym_a_coords[0]) / (
            flyvis_b_coords[0] - flyvis_a_coords[0]
        )
        row_b = flygym_a_coords[0] - row_k * flyvis_a_coords[0]
        col_k = (flygym_b_coord[1] - flygym_a_coords[1]) / (
            flyvis_b_coords[1] - flyvis_a_coords[1]
        )
        col_b = flygym_a_coords[1] - col_k * flyvis_a_coords[1]

        # Convert BoxEye.receptor_centers to flygym coordinates
        flygym_receptor_centers = np.empty_like(receptor_centers)
        flygym_receptor_centers[:, 0] = row_k * receptor_centers[:, 0] + row_b
        flygym_receptor_centers[:, 1] = col_k * receptor_centers[:, 1] + col_b

        # Get mapping FROM flyvis IDs TO flygym IDs
        self._idx_flyvis_to_flygym = (
            np.array(
                [retina.ommatidia_id_map[r, c] for r, c in flygym_receptor_centers]
            )
            - 1
        )
        self._idx_flygym_to_flyvis = np.argsort(self._idx_flyvis_to_flygym)

    def flygym_to_flyvis(self, flygym_stimulus: np.ndarray) -> np.ndarray:
        """
        Convert a stimulus from flygym's ``Retina`` representation to
        flyvision's ``BoxEye`` representation.

        Parameters
        ----------
        flygym_stimulus : np.ndarray
            Any value (e.g., intensities, neural activities) associated
            with the ommatidia in flygym's ``Retina`` ordering. The shape
            is (..., num_ommatidia): in other words, this method works as
            long as the size along the last dimension is the same as the
            number of ommatidia (721).

        Returns
        -------
        np.ndarray
            The same values, but now in flyvision's ``BoxEye`` ordering.
        """
        return flygym_stimulus[..., self._idx_flyvis_to_flygym]

    def flyvis_to_flygym(self, flyvis_stimulus: np.ndarray) -> np.ndarray:
        """
        Convert a stimulus from flyvision's ``BoxEye`` representation to
        flygym's ``Retina`` representation.

        Parameters
        ----------
        flyvis_stimulus : np.ndarray
            Any value (e.g., intensities, neural activities) associated
            with the ommatidia in flyvision's ``BoxEye`` ordering. The
            shape is (..., num_ommatidia): in other words, this method
            works as long as the size along the last dimension is the same
            as the number of ommatidia (721).

        Returns
        -------
        np.ndarray
            The same values, but now in flygym's ``Retina`` ordering.
        """
        return flyvis_stimulus[..., self._idx_flygym_to_flyvis]
