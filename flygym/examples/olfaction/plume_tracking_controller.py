import numpy as np
from enum import Enum


class WalkingState(Enum):
    FORWARD = 0, "forward"
    TURN_LEFT = 1, "left turn"
    TURN_RIGHT = 2, "right turn"
    STOP = 3, "stop"


class TurningObjective(Enum):
    UPWIND = 0
    DOWNWIND = 1


class PlumeNavigationController:
    """
    This class implements the plume navigation controller described in
    `Demir et al., 2020`_. The controller decides the fly's walking state based on the
    encounters with the plume. The controller has three states: forward walking,
    turning, and stopping. The transition among these states are governed by Poisson
    processes with encounter-dependent rates.

    .. _Demir et al., 2020: https://doi.org/10.7554/eLife.57524
    """

    def __init__(
        self,
        dt: float,
        forward_dn_drive: tuple[float, float] = (1.0, 1.0),
        left_turn_dn_drive: tuple[float, float] = (-0.4, 1.2),
        right_turn_dn_drive: tuple[float, float] = (1.2, -0.4),
        stop_dn_drive: tuple[float, float] = (0.0, 0.0),
        turn_duration: float = 0.25,  # 0.3,
        lambda_ws_0: float = 0.78,  # s^-1
        delta_lambda_ws: float = -0.8,  # -0.61,  # s^-1
        tau_s: float = 0.2,  # s
        alpha: float = 0.8,  # 0.6,  # 0.242,  # Hz^-1
        tau_freq_conv: float = 2,  # s
        cumulative_evidence_window: float = 2.0,  # s
        lambda_sw_0: float = 0.5,  # 0.29,  # s^-1
        delta_lambda_sw: float = 1,  # 0.41,  # s^-1
        tau_w=0.52,  # s
        lambda_turn: float = 1.33,  # s^-1
        random_seed: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        dt : float
            Time step of the physics simulation in seconds.
        forward_dn_drive : tuple[float, float]
            Drive values for forward walking.
        left_turn_dn_drive : tuple[float, float]
            Drive values for left turn.
        right_turn_dn_drive : tuple[float, float]
            Drive values for right turn.
        stop_dn_drive : tuple[float, float]
            Drive values for stopping.
        turn_duration : float
            Duration of the turn in seconds.
        lambda_ws_0 : float
            Baseline rate of transition from walking to stopping.
        delta_lambda_ws : float
            Change in the rate of transition from walking to stopping after an
            encounter.
        tau_s : float
            Time constant for the transition from walking to stopping.
        alpha : float
            Parameter for the sigmoid function that determines the turning direction.
        tau_freq_conv : float
            Time constant for the exponential kernel that convolves the encounter
            history to determine the turning direction.
        cumulative_evidence_window : float
            Window size for the cumulative evidence of the encounter history. In other
            words, encounters more than this many seconds ago are ignored.
        lambda_sw_0 : float
            Baseline rate of transition from stopping to walking.
        delta_lambda_sw : float
            Maximum change in the rate of transition from stopping to walking after an
            encounter.
        tau_w : float
            Time constant for evidence accumulation for the transition from stopping to
            walking.
        lambda_turn : float
            Poisson rate for turning.
        random_seed : int
            Random seed.

        Notes
        -----
        See `Demir et al., 2020`_ for details.

        .. _Demir et al., 2020: https://doi.org/10.7554/eLife.57524
        """
        self.dt = dt
        # DN drives
        self.dn_drives = {
            WalkingState.FORWARD: np.array(forward_dn_drive),
            WalkingState.TURN_LEFT: np.array(left_turn_dn_drive),
            WalkingState.TURN_RIGHT: np.array(right_turn_dn_drive),
            WalkingState.STOP: np.array(stop_dn_drive),
        }

        # Walking->stopping transition parameters
        self.lambda_ws_0 = lambda_ws_0
        self.delta_lambda_ws = delta_lambda_ws
        self.tau_s = tau_s

        self.curr_time = 0.0
        self.curr_state = WalkingState.FORWARD
        self.curr_state_start_time = 0.0
        self.last_encounter_time = -np.inf
        self.encounter_history = []

        # Stopping->walking transition parameters
        self.cumulative_evidence_window = cumulative_evidence_window
        self.cumulative_evidence_len = int(cumulative_evidence_window / dt)
        self.lambda_sw_0 = lambda_sw_0
        self.delta_lambda_sw = delta_lambda_sw
        self.tau_w = tau_w
        self.encounter_weights = (
            -np.arange(self.cumulative_evidence_len)[::-1] * self.dt
        )

        # Turning related parameters
        self.turn_duration = turn_duration
        self.alpha = alpha
        self.tau_freq_conv = tau_freq_conv
        self.freq_kernel = np.exp(self.encounter_weights / tau_freq_conv)
        self.lambda_turn = lambda_turn

        self._turn_debug_str_buffer = ""
        self.random_state = np.random.RandomState(random_seed)

    def decide_state(self, encounter_flag: bool, fly_heading: np.ndarray):
        """
        Decide the fly's walking state based on the encounter information. If the next
        state is turning, the turning direction is further determined based on the
        encounter frequency and the fly's current heading (upwind or downwind).
        """
        self.encounter_history.append(encounter_flag)
        if encounter_flag:
            self.last_encounter_time = self.curr_time

        debug_str = ""

        # Forward -> turn transition
        if self.curr_state == WalkingState.FORWARD:
            p_nochange = np.exp(-self.lambda_turn * self.dt)
            if self.random_state.rand() > p_nochange:
                encounter_hist = np.array(
                    self.encounter_history[-self.cumulative_evidence_len :]
                )
                kernel = self.freq_kernel[-len(encounter_hist) :]
                w_freq = np.sum(kernel * encounter_hist) * self.dt
                correction_factor = self.exp_integral_norm_factor(
                    window=len(encounter_hist) * self.dt, tau=self.tau_freq_conv
                )
                w_freq *= correction_factor
                p_upwind = 1 / (1 + np.exp(-self.alpha * w_freq))
                if self.random_state.rand() < p_upwind:
                    turn_objective = TurningObjective.UPWIND
                    debug_str = (
                        f"Wfreq={w_freq:.2f}  "
                        f"P(upwind)={p_upwind:.2f}, turning UPWIND"
                    )
                else:
                    turn_objective = TurningObjective.DOWNWIND
                    debug_str = (
                        f"Wfreq={w_freq:.2f}  "
                        f"P(upwind)={p_upwind:.2f}, turning DOWNWIND"
                    )
                self._turn_debug_str_buffer = debug_str

                if fly_heading[1] >= 0:  # upwind == left turn
                    if turn_objective == TurningObjective.UPWIND:
                        self.curr_state = WalkingState.TURN_LEFT
                    else:
                        self.curr_state = WalkingState.TURN_RIGHT
                else:
                    if turn_objective == TurningObjective.UPWIND:
                        self.curr_state = WalkingState.TURN_RIGHT
                    else:
                        self.curr_state = WalkingState.TURN_LEFT
                self.curr_state_start_time = self.curr_time

        # Forward -> stop transition
        if self.curr_state == WalkingState.FORWARD:
            lambda_ws = self.lambda_ws_0 + self.delta_lambda_ws * np.exp(
                -(self.curr_time - self.last_encounter_time) / self.tau_s
            )
            p_nochange = np.exp(-lambda_ws * self.dt)
            p_stop_1s = 1 - np.exp(-lambda_ws)
            debug_str = (
                f"lambda(w->s)={lambda_ws:.2f}  P(stop within 1s)={p_stop_1s:.2f}"
            )
            if self.random_state.rand() > p_nochange:
                self.curr_state = WalkingState.STOP
                self.curr_state_start_time = self.curr_time

        # Turn -> forward transition
        if self.curr_state in (WalkingState.TURN_LEFT, WalkingState.TURN_RIGHT):
            debug_str = self._turn_debug_str_buffer
            if self.curr_time - self.curr_state_start_time > self.turn_duration:
                self.curr_state = WalkingState.FORWARD
                self.curr_state_start_time = self.curr_time

        # Stop -> forward transition
        if self.curr_state == WalkingState.STOP:
            encounter_hist = np.array(
                self.encounter_history[-self.cumulative_evidence_len :]
            )
            time_diff = self.encounter_weights[-len(encounter_hist) :]
            cumm_evidence_integral = (
                np.sum(np.exp(time_diff / self.tau_w) * encounter_hist) * self.dt
            )
            correction_factor = self.exp_integral_norm_factor(
                window=len(encounter_hist) * self.dt, tau=self.tau_w
            )
            # print("s->w", correction_factor)
            cumm_evidence_integral *= correction_factor
            lambda_sw = self.lambda_sw_0 + self.delta_lambda_sw * cumm_evidence_integral
            p_nochange = np.exp(-lambda_sw * self.dt)
            p_walk_1s = 1 - np.exp(-lambda_sw)
            debug_str = (
                f"lambda(s->w)={lambda_sw:.2f}  P(walk within 1s)={p_walk_1s:.2f}"
            )
            if self.curr_time > 2:
                pass
            if self.random_state.rand() > p_nochange:
                self.curr_state = WalkingState.FORWARD
                self.curr_state_start_time = self.curr_time

        self.curr_time += self.dt
        return self.curr_state, self.dn_drives[self.curr_state], debug_str

    def exp_integral_norm_factor(self, window: float, tau: float):
        r"""
        In case the exponential kernel is truncated to a finite length, this method
        computes a scaler k(w) that correct the underestimation of the integrated value:

        .. math::
            k(w) =
                \frac{\int_{-\infty}^0 e^{t / \tau} dt}
                    {\int_{-w}^0 e^{t / \tau} dt}
            = \frac{1}{1 - e^{-w/\tau}}

        Parameters
        ----------
        window : float
            Window size for cumulative evidence in seconds.
        tau : float
            Time scale for the exponential kernel.

        Returns
        -------
        float
            The correction factor.
        """
        if window <= 0:
            raise ValueError("Window size must be positive for cumulative evidence")
        return 1 / (1 - np.exp(-window / tau))
