import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from tqdm import trange

import flygym.mujoco
from flygym.common import get_data_path


class PreprogrammedSteps:
    """Preprogrammed steps by each leg extracted from experimental
    recordings.

    Attributes
    ----------
    legs : List[str]
        List of leg names (eg. LF for left front leg).
    dofs_per_leg : List[str]
        List of names for degrees of freedom for each leg.
    duration : float
        Duration of the preprogrammed step (at 1x speed) in seconds.
    neutral_pos : Dict[str, np.ndarray]
        Neutral position of DoFs for each leg. Keys are leg names; values
        are joint angles in the order of ``self.dofs_per_leg``.
    swing_period : Dict[str, np.ndarray]
        The start and end of the liftedswing phase for each leg. Keys are
        leg names; values are arrays of shape (2,) with the start and end
        of the swing normalized to [0, 2π].

    Parameters
    ----------
    path : str or Path, optional
        Path to the preprogrammed steps data. If None, the default
        preprogrammed steps data will be loaded.
    """

    legs = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
    dofs_per_leg = [
        "Coxa",
        "Coxa_roll",
        "Coxa_yaw",
        "Femur",
        "Femur_roll",
        "Tibia",
        "Tarsus1",
    ]

    def __init__(self, path=None):
        if path is None:
            path = get_data_path("flygym", "data") / "behavior/single_steps.pkl"
        with open(path, "rb") as f:
            single_steps_data = pickle.load(f)
        self._length = len(single_steps_data["joint_LFCoxa"])
        self._timestep = single_steps_data["meta"]["timestep"]
        self.duration = self._length * self._timestep

        phase_grid = np.linspace(0, 2 * np.pi, self._length)
        self._psi_funcs = {}
        for leg in self.legs:
            joint_angles = np.array(
                [single_steps_data[f"joint_{leg}{dof}"] for dof in self.dofs_per_leg]
            )
            self._psi_funcs[leg] = CubicSpline(
                phase_grid, joint_angles, axis=1, bc_type="periodic"
            )

        self.neutral_pos = {
            leg: self._psi_funcs[leg](0)[:, np.newaxis] for leg in self.legs
        }

        swing_stance_time_dict = single_steps_data["swing_stance_time"]
        self.swing_period = {}
        for leg in self.legs:
            my_swing_period = np.array([0, swing_stance_time_dict["stance"][leg]])
            my_swing_period /= self.duration
            my_swing_period *= 2 * np.pi
            self.swing_period[leg] = my_swing_period
        self._swing_start_arr = np.array(
            [self.swing_period[leg][0] for leg in self.legs]
        )
        self._swing_end_arr = np.array([self.swing_period[leg][1] for leg in self.legs])

    def get_joint_angles(self, leg, phase, magnitude=1):
        """Get joint angles for a given leg at a given phase.

        Parameters
        ----------
        leg : str
            Leg name.
        phase : float or np.ndarray
            Phase or array of phases of the step normalized to [0, 2π].
        magnitude : float or np.ndarray, optional
            Magnitude of the step. Default: 1 (the preprogrammed steps as
            provided).

        Returns
        -------
        np.ndarray
            Joint angles of the leg at the given phase(s). The shape of the
            array is (7, n) if ``phase`` is a 1D array of n elements, or
            (7,) if ``phase`` is a scalar.
        """
        if isinstance(phase, float) or phase.shape == ():
            phase = np.array([phase])
        psi_func = self._psi_funcs[leg]
        offset = psi_func(phase) - self.neutral_pos[leg]
        joint_angles = self.neutral_pos[leg] + magnitude * offset
        return joint_angles.squeeze()

    def get_adhesion_onoff(self, leg, phase):
        """Get whether adhesion is on for a given leg at a given phase.

        Parameters
        ----------
        leg : str
            Leg name.
        phase : float or np.ndarray
            Phase or array of phases of the step normalized to [0, 2π].

        Returns
        -------
        bool or np.ndarray
            Whether adhesion is on for the leg at the given phase(s).
            A boolean array of shape (n,) is returned if ``phase`` is a 1D
            array of n elements; a bool is returned if ``phase`` is a
            scalar.
        """
        swing_start, swing_end = self.swing_period[leg]
        return not (swing_start < phase % (2 * np.pi) < swing_end)


class RuleBasedSteppingCoordinator:
    legs = ["LF", "LM", "LH", "RF", "RM", "RH"]

    def __init__(
        self, timestep, rules_graph, weights, preprogrammed_steps, margin=0.001, seed=0
    ):
        self.timestep = timestep
        self.rules_graph = rules_graph
        self.weights = weights
        self.preprogrammed_steps = preprogrammed_steps
        self.margin = margin
        self.random_state = np.random.RandomState(seed)
        self._phase_inc_per_step = (
            2 * np.pi * (timestep / self.preprogrammed_steps.duration)
        )
        self.curr_step = 0

        self.rule1_scores = np.zeros(6)
        self.rule2_scores = np.zeros(6)
        self.rule3_scores = np.zeros(6)

        self.leg_phases = np.zeros(6)
        self.mask_is_stepping = np.zeros(6, dtype=bool)

        self._leg2id = {leg: i for i, leg in enumerate(self.legs)}
        self._id2leg = {i: leg for i, leg in enumerate(self.legs)}

    @property
    def combined_scores(self):
        return self.rule1_scores + self.rule2_scores + self.rule3_scores

    def _get_eligible_legs(self):
        score_thr = self.combined_scores.max()
        score_thr = max(0, score_thr - np.abs(score_thr) * self.margin)
        mask_is_eligible = (
            (self.combined_scores >= score_thr)  # highest or almost highest score
            & (self.combined_scores > 0)  # score is positive
            & ~self.mask_is_stepping  # leg is not currently stepping
        )
        return np.where(mask_is_eligible)[0]

    def _select_stepping_leg(self):
        eligible_legs = self._get_eligible_legs()
        if len(eligible_legs) == 0:
            return None
        return self.random_state.choice(eligible_legs)

    def _apply_rule1(self):
        for i, leg in enumerate(self.legs):
            is_swinging = (
                0 < self.leg_phases[i] < self.preprogrammed_steps.swing_period[leg][1]
            )
            edges = filter_edges(self.rules_graph, "rule1", src_node=leg)
            for _, tgt in edges:
                self.rule1_scores[self._leg2id[tgt]] = (
                    self.weights["rule1"] if is_swinging else 0
                )

    def _get_stance_progress_ratio(self, leg):
        swing_start, swing_end = self.preprogrammed_steps.swing_period[leg]
        stance_duration = 2 * np.pi - swing_end
        curr_stance_progress = self.leg_phases[self._leg2id[leg]] - swing_end
        curr_stance_progress = max(0, curr_stance_progress)
        return curr_stance_progress / stance_duration

    def _apply_rule2(self):
        self.rule2_scores[:] = 0
        for i, leg in enumerate(self.legs):
            stance_progress_ratio = self._get_stance_progress_ratio(leg)
            if stance_progress_ratio == 0:
                continue
            for side in ["ipsi", "contra"]:
                edges = filter_edges(self.rules_graph, f"rule2_{side}", src_node=leg)
                weight = self.weights[f"rule2_{side}"]
                for _, tgt in edges:
                    tgt_id = self._leg2id[tgt]
                    self.rule2_scores[tgt_id] += weight * (1 - stance_progress_ratio)

    def _apply_rule3(self):
        self.rule3_scores[:] = 0
        for i, leg in enumerate(self.legs):
            stance_progress_ratio = self._get_stance_progress_ratio(leg)
            if stance_progress_ratio == 0:
                continue
            for side in ["ipsi", "contra"]:
                edges = filter_edges(self.rules_graph, f"rule3_{side}", src_node=leg)
                weight = self.weights[f"rule3_{side}"]
                for _, tgt in edges:
                    tgt_id = self._leg2id[tgt]
                    self.rule3_scores[tgt_id] += weight * stance_progress_ratio

    def step(self):
        if self.curr_step == 0:
            # The first step is always a fore leg or mid leg
            stepping_leg_id = self.random_state.choice([0, 1, 3, 4])
        else:
            stepping_leg_id = self._select_stepping_leg()

        # Initiate a new step, if conditions are met for any leg
        if stepping_leg_id is not None:
            self.mask_is_stepping[stepping_leg_id] = True  # start stepping this leg

        # Progress all stepping legs
        self.leg_phases[self.mask_is_stepping] += self._phase_inc_per_step

        # Check if any stepping legs has completed a step
        mask_has_newly_completed = self.leg_phases >= 2 * np.pi
        self.leg_phases[mask_has_newly_completed] = 0
        self.mask_is_stepping[mask_has_newly_completed] = False

        # Update scores
        self._apply_rule1()
        self._apply_rule2()
        self._apply_rule3()

        self.curr_step += 1


def filter_edges(graph, rule, src_node=None):
    """Return a list of edges that match the given rule and source node.
    The edges are returned as a list of tuples (src, tgt)."""
    return [
        (src, tgt)
        for src, tgt, rule_type in graph.edges(data="rule")
        if (rule_type == rule) and (src_node is None or src == src_node)
    ]


def plot_time_series_multi_legs(
    time_series_block,
    timestep,
    spacing=10,
    legs=["LF", "LM", "LH", "RF", "RM", "RH"],
    ax=None,
):
    """Plot a time series of scores for multiple legs.

    Parameters
    ----------
    time_series_block : np.ndarray
        Time series of scores for multiple legs. The shape of the array
        should be (n, m), where n is the number of time steps and m is the
        length of ``legs``.
    timestep : float
        Timestep of the time series in seconds.
    spacing : float, optional
        Spacing between the time series of different legs. Default: 10.
    legs : List[str], optional
        List of leg names. Default: ["LF", "LM", "LH", "RF", "RM", "RH"].
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    t_grid = np.arange(time_series_block.shape[0]) * timestep
    spacing *= -1
    offset = np.arange(6)[np.newaxis, :] * spacing
    score_hist_viz = time_series_block + offset
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
    for i in range(len(legs)):
        ax.axhline(offset.flatten()[i], color="k", linewidth=0.5)
        ax.plot(t_grid, score_hist_viz[:, i])
    ax.set_yticks(offset[0], legs)
    ax.set_xlabel("Time (s)")
    return ax


def construct_rules_graph():
    # For each rule, the keys are the source nodes and the values are the
    # target nodes influenced by the source nodes
    edges = {
        "rule1": {"LM": ["LF"], "LH": ["LM"], "RM": ["RF"], "RH": ["RM"]},
        "rule2": {
            "LF": ["RF"],
            "LM": ["RM", "LF"],
            "LH": ["RH", "LM"],
            "RF": ["LF"],
            "RM": ["LM", "RF"],
            "RH": ["LH", "RM"],
        },
        "rule3": {
            "LF": ["RF", "LM"],
            "LM": ["RM", "LH"],
            "LH": ["RH"],
            "RF": ["LF", "RM"],
            "RM": ["LM", "RH"],
            "RH": ["LH"],
        },
    }

    # Construct the rules graph
    rules_graph = nx.MultiDiGraph()
    for rule_type, d in edges.items():
        for src, tgt_nodes in d.items():
            for tgt in tgt_nodes:
                if rule_type == "rule1":
                    rule_type_detailed = rule_type
                else:
                    side = "ipsi" if src[0] == tgt[0] else "contra"
                    rule_type_detailed = f"{rule_type}_{side}"
                rules_graph.add_edge(src, tgt, rule=rule_type_detailed)

    return rules_graph


if __name__ == "__main__":
    run_time = 1
    timestep = 1e-4
    weights = {
        "rule1": -10,
        "rule2_ipsi": 2.5,
        "rule2_contra": 1,
        "rule3_ipsi": 3.0,
        "rule3_contra": 2.0,
    }

    preprogrammed_steps = PreprogrammedSteps()
    rules_graph = construct_rules_graph()
    controller = RuleBasedSteppingCoordinator(
        timestep=timestep,
        rules_graph=rules_graph,
        weights=weights,
        preprogrammed_steps=preprogrammed_steps,
    )
    sim_params = flygym.mujoco.Parameters(
        timestep=timestep,
        render_mode="saved",
        render_playspeed=0.1,
        enable_adhesion=True,
        draw_adhesion=True,
    )
    nmf = flygym.mujoco.NeuroMechFly(
        sim_params=sim_params,
        init_pose="stretch",
        actuated_joints=flygym.mujoco.preprogrammed.all_leg_dofs,
        control="position",
    )

    obs, info = nmf.reset()
    for i in trange(int(run_time / sim_params.timestep)):
        controller.step()
        joint_angles = []
        adhesion_onoff = []
        for leg, phase in zip(controller.legs, controller.leg_phases):
            joint_angles_arr = preprogrammed_steps.get_joint_angles(leg, phase)
            joint_angles.append(joint_angles_arr.flatten())
            adhesion_onoff.append(preprogrammed_steps.get_adhesion_onoff(leg, phase))
        action = {
            "joints": np.concatenate(joint_angles),
            "adhesion": np.array(adhesion_onoff),
        }
        obs, reward, terminated, truncated, info = nmf.step(action)
        nmf.render()

    nmf.save_video("./outputs/rule_based_controller.mp4")
