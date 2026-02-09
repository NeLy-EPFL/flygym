from dataclasses import dataclass

__all__ = ["ContactParams"]


@dataclass(kw_only=True)
class ContactParams:
    """Contact constraint parameters for MuJoCo physics simulation.

    This class bundles friction coefficients, solver reference acceleration, and solver
    impedance settings used to enforce contact constraints between colliding bodies.

    **Solver reference acceleration** defines how quickly constraint violations (e.g.,
    body penetration) are corrected.

    **Solver impedance** determines how hard or soft the contact constraint is by
    controlling the impedance (inverse of compliance). Higher impedance creates stiffer
    contacts that resist penetration more strongly, while lower impedance allows more
    compliance. The impedance can vary with penetration depth to model contacts that
    become stiffer as penetration increases.

    See `MuJoCo documentation <https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters>`_
    for details.

    Attributes:
        sliding_friction:
            Tangential friction coefficient. Default: 5.0 (MuJoCo default: 1.0).
        torsional_friction:
            Torsional friction coefficient. Default: 0.02 (MuJoCo default: 0.005).
        rolling_friction:
            Rolling friction coefficient. Default: 0.0001 (MuJoCo default: 0.0001).
        solver_refaccl_timeconst:
            Time constant for constraint correction. Lower values resolve violations
            faster. Default: 0.0002 (MuJoCo default: 0.02).
        solver_refaccl_dampratio:
            Damping ratio for constraint correction (positive). <1: underdamped,
            1: critically damped, >1: overdamped. Default: 1.0 (MuJoCo default: 1.0).
        solver_impedance_min:
            Minimum constraint impedance at initial contact, range (0, 1). Applied upon
            "first contact". Default: 0.98 (MuJoCo default: 0.9).
        solver_impedance_max:
            Maximum constraint impedance at deep penetration, range (0, 1). Must be
            greater than or equal to min. Applied when penetration exceeds the
            transition width defined below. Default: 0.99 (MuJoCo default: 0.95).
        solver_impedance_min2max_width:
            Penetration distance over which impedance transitions from min to max.
            Default: 1e-5 (MuJoCo default: 1e-3).
        solver_impedance_transitionmidpoint:
            Relative position (0, 1) within the transition width where "mid-range" force
            is applied. Default: 0.5 (MuJoCo default: 0.5).
        solver_impedance_transitionsharpness:
            Controls transition curve shape (>=1). 1 is linear, higher values reflect
            sharper min-to-max. Default: 3.0 (MuJoCo default: 2.0).
        margin:
            Contact force starts to be generated from this distance before actual
            contact. This is helpful for preventing tiny leg tips from penetrating the
            ground. Default: 1e-3 (MuJoCo default: 0).
    """

    # ===== Contact friction =====
    sliding_friction: float = 1.0
    torsional_friction: float = 2e-2
    rolling_friction: float = 1e-4

    # ===== Contact solver reference acceleration =====
    solver_refaccl_timeconst: float = 2e-4
    solver_refaccl_dampratio: float = 1.0

    # ===== Contact solver impedance =====
    solver_impedance_min: float = 0.98
    solver_impedance_max: float = 0.99
    solver_impedance_min2max_width: float = 1e-5
    solver_impedance_transitionmidpoint: float = 0.5
    solver_impedance_transitionsharpness: float = 3.0

    # ===== Geometric margin =====
    margin: float = 1e-3

    def get_friction_tuple(self):
        """Return the MuJoCo `friction` parameter for contact pairs. Note that MuJoCo
        expects five coefficients for explicitly `contact/pair` objects (2x sliding,
        1x torsional, 2x rolling). This differs from the `friction` attribute of
        `body/geom` objects, which only has three coefficients (the 5-tuple is
        determined automatically depending on the two colliding geometries)."""
        self._raise_on_invalid_friction()
        # For contact between two geometries: 2 tangential, 1 torsional, 2 rolling
        return (
            self.sliding_friction,
            self.sliding_friction,
            self.torsional_friction,
            self.rolling_friction,
            self.rolling_friction,
        )

    def get_solref_tuple(self):
        """Return the MuJoCo `solref` parameter for contact pairs."""
        self._raise_on_invalid_solver_refaccl()
        return (
            self.solver_refaccl_timeconst,
            self.solver_refaccl_dampratio,
        )

    def get_solimp_tuple(self):
        """Return the MuJoCo `solimp` parameter for contact pairs."""
        self._raise_on_invalid_solver_impedance()
        return (
            self.solver_impedance_min,
            self.solver_impedance_max,
            self.solver_impedance_transitionmidpoint,
            self.solver_impedance_transitionsharpness,
        )

    def is_valid(self, raise_on_invalid=True):
        try:
            self._raise_on_invalid_friction()
            self._raise_on_invalid_solver_refaccl()
            self._raise_on_invalid_solver_impedance()
            return True
        except ValueError as e:
            if raise_on_invalid:
                raise ValueError(f"Invalid ContactParams: {e}") from e
            return False

    def _raise_on_invalid_friction(self):
        if not (self.sliding_friction >= 0):
            raise ValueError("Sliding friction must be non-negative")
        if not (self.torsional_friction >= 0):
            raise ValueError("Torsional friction must be non-negative")
        if not (self.rolling_friction >= 0):
            raise ValueError("Rolling friction must be non-negative")

    def _raise_on_invalid_solver_refaccl(self):
        if not (self.solver_refaccl_timeconst > 0):
            raise ValueError("Solver reference time constant must be positive")
        if not (self.solver_refaccl_dampratio > 0):
            raise ValueError("Solver reference damping ratio must be positive")

    def _raise_on_invalid_solver_impedance(self):
        if not (0 < self.solver_impedance_min < 1):
            raise ValueError("Minimum solver impedance must be in (0, 1)")
        if not (0 < self.solver_impedance_max < 1):
            raise ValueError("Maximum solver impedance must be in (0, 1)")
        if not (self.solver_impedance_max >= self.solver_impedance_min):
            raise ValueError("Maximum solver impedance cannot be less than minimum")
        if not (self.solver_impedance_min2max_width > 0):
            raise ValueError(
                "Impedance mid-to-max transition must happen over a positive distance"
            )
        if not (0 < self.solver_impedance_transitionmidpoint < 1):
            raise ValueError("Midpoint of impedance min-to-max must be in (0, 1)")
        if not (self.solver_impedance_transitionsharpness >= 1):
            raise ValueError(
                "Sharpness of impedance transition must be at least linear (1)"
            )
