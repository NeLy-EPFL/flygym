import numpy as np
import networkx as nx
from tqdm import trange

from flygym.examples.locomotion import PreprogrammedSteps


class RuleBasedController:
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
        """
        Parameters
        ----------
        timestep : float
            The timestep of the simulation.
        rules_graph : nx.MultiDiGraph
            The rules graph that defines the interactions between the legs.
        weights : dict
            The weights for each rule.
        preprogrammed_steps : PreprogrammedSteps, optional
            Preprogrammed steps to be used for leg movement.
        margin : float, optional
            The margin for selecting the highest scoring leg.
        seed : int, optional
            The random seed to use for selecting the highest scoring leg.
        """
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
        """
        The global score for all legs. The highest scoring leg is
        selected for stepping.
        """
        return self.rule1_scores + self.rule2_scores + self.rule3_scores

    def _get_eligible_legs(self):
        """
        Return a mask of legs that are eligible for stepping (i.e. the score is:
        the highest within margin, above zeros and the leg is not currently stepping).
        """
        # compute the score threshold as (1.0 - margin)*100% of the highest score
        score_thr = self.combined_scores.max()
        score_thr = max(0, score_thr - np.abs(score_thr) * self.margin)
        mask_is_eligible = (
            (self.combined_scores >= score_thr)  # highest or almost highest score
            & (self.combined_scores > 0)  # score is positive
            & ~self.mask_is_stepping  # leg is not currently stepping
        )
        return np.where(mask_is_eligible)[0]

    def _select_stepping_leg(self):
        """
        Select the leg to be stepped within the eligible legs.
        The choice is random if multiple legs have a high score.
        """
        eligible_legs = self._get_eligible_legs()
        if len(eligible_legs) == 0:
            return None
        return self.random_state.choice(eligible_legs)

    def _apply_rule1(self):
        """
        Compute the contribution of rule 1 to the scores.
        Rule 1: A leg is less likely to step if the connected legs are in stance .
        """
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
        """
        Compute the progress of the stance phase for the given leg.
        0 if the leg just started the stance phase,
        0.5 if the leg is halfway through the stance and
        1 if the leg is about to swing.
        """
        swing_start, swing_end = self.preprogrammed_steps.swing_period[leg]
        stance_duration = 2 * np.pi - swing_end
        curr_stance_progress = self.leg_phases[self._leg2id[leg]] - swing_end
        curr_stance_progress = max(0, curr_stance_progress)
        return curr_stance_progress / stance_duration

    def _apply_rule2(self):
        """
        Compute the contribution of rule 2 to the scores.
        Rule 2: Anterior legs are more likely to step
        if the posterior leg is early in the stance phase.
        """
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
        """
        Compute the contribution of rule 3 to the scores.
        Rule 3: Posterior legs are more likely to step
        if the anterior leg is late in the stance phase.
        """
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
        """
        Steps the controller for one timestep.
        Updates the leg phases.
        Updates the scores based on the rules.
        """
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


def construct_rules_graph():
    """Construct a rules graph that defines the interactions between the legs.
    The rules graph is a directed multigraph where the nodes are the legs and the
    edges are the rules that define the interactions between the legs."""

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


def run_rule_based_simulation(sim, controller, run_time, pbar=True):
    obs, info = sim.reset()
    obs_list = []
    range_ = trange if pbar else range
    for _ in range_(int(run_time / sim.timestep)):
        controller.step()
        joint_angles = []
        adhesion_onoff = []
        for leg, phase in zip(controller.legs, controller.leg_phases):
            joint_angles_arr = controller.preprogrammed_steps.get_joint_angles(
                leg, phase
            )
            joint_angles.append(joint_angles_arr.flatten())
            adhesion_onoff.append(
                controller.preprogrammed_steps.get_adhesion_onoff(leg, phase)
            )
        action = {
            "joints": np.concatenate(joint_angles),
            "adhesion": np.array(adhesion_onoff),
        }
        obs, reward, terminated, truncated, info = sim.step(action)
        obs_list.append(obs)
        sim.render()

    return obs_list


if __name__ == "__main__":
    from flygym import Fly, Camera, SingleFlySimulation
    from flygym.preprogrammed import all_leg_dofs

    run_time = 1
    timestep = 1e-4

    # Initialize preprogrammed steps
    preprogrammed_steps = PreprogrammedSteps()

    # Initialize rule-based controller
    weights = {
        "rule1": -10,
        "rule2_ipsi": 2.5,
        "rule2_contra": 1,
        "rule3_ipsi": 3.0,
        "rule3_contra": 2.0,
    }
    rules_graph = construct_rules_graph()
    controller = RuleBasedController(
        timestep=timestep,
        rules_graph=rules_graph,
        weights=weights,
        preprogrammed_steps=preprogrammed_steps,
    )

    # Initialize NeuroMechFly simulation
    fly = Fly(
        init_pose="stretch",
        actuated_joints=all_leg_dofs,
        control="position",
        enable_adhesion=True,
        draw_adhesion=True,
    )

    cam = Camera(
        fly=fly,
        play_speed=0.1,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
    )

    # Run simulation
    run_rule_based_simulation(sim, controller, run_time)

    # Save video
    cam.save_video("./outputs/rule_based_controller.mp4")
