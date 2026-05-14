from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flygym.anatomy import JointDOF, LEGS
from flygym.examples.locomotion.common import LocomotionAction
from flygym.examples.locomotion.preprogrammed import PreprogrammedSteps

RuleGraph = dict[str, dict[str, tuple[str, ...]]]


def construct_rules_graph() -> RuleGraph:
    """Return the Walknet rule graph used by the v1 rule-based tutorial."""
    return {
        "rule1": {"lm": ("lf",), "lh": ("lm",), "rm": ("rf",), "rh": ("rm",)},
        "rule2": {
            "lf": ("rf",),
            "lm": ("rm", "lf"),
            "lh": ("rh", "lm"),
            "rf": ("lf",),
            "rm": ("lm", "rf"),
            "rh": ("lh", "rm"),
        },
        "rule3": {
            "lf": ("rf", "lm"),
            "lm": ("rm", "lh"),
            "lh": ("rh",),
            "rf": ("lf", "rm"),
            "rm": ("lm", "rh"),
            "rh": ("lh",),
        },
    }


@dataclass
class RuleBasedController:
    """Local sensory-rule leg coordinator ported from the v1 Walknet tutorial."""

    timestep: float
    rules_graph: RuleGraph | None = None
    weights: dict[str, float] | None = None
    preprogrammed_steps: PreprogrammedSteps = field(default_factory=PreprogrammedSteps)
    output_dof_order: list[JointDOF] | None = None
    margin: float = 0.001
    seed: int = 0

    legs: tuple[str, ...] = tuple(LEGS)

    def __post_init__(self) -> None:
        if self.rules_graph is None:
            self.rules_graph = construct_rules_graph()
        if self.weights is None:
            self.weights = {
                "rule1": -10.0,
                "rule2_ipsi": 2.5,
                "rule2_contra": 1.0,
                "rule3_ipsi": 3.0,
                "rule3_contra": 2.0,
            }
        self.random_state = np.random.RandomState(self.seed)
        self._phase_inc_per_step = (
            2 * np.pi * (self.timestep / self.preprogrammed_steps.duration)
        )
        self.curr_step = 0
        self.rule1_scores = np.zeros(6)
        self.rule2_scores = np.zeros(6)
        self.rule3_scores = np.zeros(6)
        self.leg_phases = np.zeros(6)
        self.mask_is_stepping = np.zeros(6, dtype=bool)
        self._leg2id = {leg: i for i, leg in enumerate(self.legs)}

    @property
    def combined_scores(self) -> np.ndarray:
        return self.rule1_scores + self.rule2_scores + self.rule3_scores

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        self.curr_step = 0
        self.rule1_scores[:] = 0
        self.rule2_scores[:] = 0
        self.rule3_scores[:] = 0
        self.leg_phases[:] = 0
        self.mask_is_stepping[:] = False

    def step(self) -> LocomotionAction:
        """Advance the coordinator and return a simulation action."""
        if self.curr_step == 0:
            stepping_leg_id = self.random_state.choice([0, 1, 3, 4])
        else:
            stepping_leg_id = self._select_stepping_leg()

        if stepping_leg_id is not None:
            self.mask_is_stepping[stepping_leg_id] = True

        self.leg_phases[self.mask_is_stepping] += self._phase_inc_per_step
        mask_has_newly_completed = self.leg_phases >= 2 * np.pi
        self.leg_phases[mask_has_newly_completed] = 0
        self.mask_is_stepping[mask_has_newly_completed] = False

        self._apply_rule1()
        self._apply_rule2()
        self._apply_rule3()
        self.curr_step += 1

        joint_angles = self.preprogrammed_steps.get_joint_angles_by_dof_order(
            self.leg_phases,
            np.ones(6),
            self.output_dof_order,
        )
        adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff_by_phase(
            self.leg_phases
        )
        return LocomotionAction(
            joint_angles=joint_angles, adhesion_onoff=adhesion_onoff
        )

    def _get_eligible_legs(self) -> np.ndarray:
        score_thr = self.combined_scores.max()
        score_thr = max(0, score_thr - np.abs(score_thr) * self.margin)
        mask_is_eligible = (
            (self.combined_scores >= score_thr)
            & (self.combined_scores > 0)
            & ~self.mask_is_stepping
        )
        return np.where(mask_is_eligible)[0]

    def _select_stepping_leg(self) -> int | None:
        eligible_legs = self._get_eligible_legs()
        if len(eligible_legs) == 0:
            return None
        return int(self.random_state.choice(eligible_legs))

    def _apply_rule1(self) -> None:
        self.rule1_scores[:] = 0
        for i, leg in enumerate(self.legs):
            is_swinging = (
                0 < self.leg_phases[i] < self.preprogrammed_steps.swing_period[leg][1]
            )
            for tgt in self.rules_graph["rule1"].get(leg, ()):
                self.rule1_scores[self._leg2id[tgt]] = (
                    self.weights["rule1"] if is_swinging else 0
                )

    def _get_stance_progress_ratio(self, leg: str) -> float:
        _, swing_end = self.preprogrammed_steps.swing_period[leg]
        stance_duration = 2 * np.pi - swing_end
        curr_stance_progress = self.leg_phases[self._leg2id[leg]] - swing_end
        curr_stance_progress = max(0, curr_stance_progress)
        return curr_stance_progress / stance_duration

    def _apply_rule2(self) -> None:
        self.rule2_scores[:] = 0
        for leg in self.legs:
            stance_progress_ratio = self._get_stance_progress_ratio(leg)
            if stance_progress_ratio == 0:
                continue
            for tgt in self.rules_graph["rule2"].get(leg, ()):
                side = "ipsi" if leg[0] == tgt[0] else "contra"
                weight = self.weights[f"rule2_{side}"]
                self.rule2_scores[self._leg2id[tgt]] += weight * (
                    1 - stance_progress_ratio
                )

    def _apply_rule3(self) -> None:
        self.rule3_scores[:] = 0
        for leg in self.legs:
            stance_progress_ratio = self._get_stance_progress_ratio(leg)
            if stance_progress_ratio == 0:
                continue
            for tgt in self.rules_graph["rule3"].get(leg, ()):
                side = "ipsi" if leg[0] == tgt[0] else "contra"
                weight = self.weights[f"rule3_{side}"]
                self.rule3_scores[self._leg2id[tgt]] += weight * stance_progress_ratio
