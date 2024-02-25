import numpy as np
import networkx as nx
from tqdm import trange

import flygym
from flygym.examples.common import PreprogrammedSteps


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


def run_rule_based_simulation(nmf, controller, run_time):
    obs, info = nmf.reset()
    for _ in trange(int(run_time / nmf.sim_params.timestep)):
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
        obs, reward, terminated, truncated, info = nmf.step(action)
        nmf.render()


if __name__ == "__main__":
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
    controller = RuleBasedSteppingCoordinator(
        timestep=timestep,
        rules_graph=rules_graph,
        weights=weights,
        preprogrammed_steps=preprogrammed_steps,
    )

    # Initialize NeuroMechFly simulation
    sim_params = flygym.Parameters(
        timestep=timestep,
        render_mode="saved",
        render_playspeed=0.1,
        enable_adhesion=True,
        draw_adhesion=True,
    )
    nmf = flygym.NeuroMechFly(
        sim_params=sim_params,
        init_pose="stretch",
        actuated_joints=flygym.preprogrammed.all_leg_dofs,
        control="position",
    )

    # Run simulation
    run_rule_based_simulation(nmf, controller, run_time)

    # Save video
    nmf.save_video("./outputs/rule_based_controller.mp4")
