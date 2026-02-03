from dataclasses import dataclass


@dataclass(kw_only=True)
class ContactParams:
    """Parameters for contact constraints. See a very detailed description at
    https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters"""

    # ===== Contact friction =====
    sliding_friction: float = 5.0  # mujoco default: 1
    torsional_friction: float = 2e-2  # MuJoCo default: 5e-3
    rolling_friction: float = 1e-4  # MuJoCo default: 1e-4

    # ===== Contact solver reference acceleration =====
    # When contact constraint is violated, the reference acceleration determines how
    # fast a motion should be applied to rectify the violation.
    # Time constant: lower = faster correction
    solver_refaccl_timeconst: float = 2e-4  # MuJoCo default: 2e-2
    # Damping ratio: <1 is underdamping, 1 is critical damping, >1 is overdamping
    solver_refaccl_dampratio: float = 1.0  # MuJoCo default: 1

    # ===== Contact solver impedance =====
    # When contact constraint is violated, the impedance determines how much force is
    # generated to correct the violation. Low impedance = soft constraint, vice versa.
    # Min impedance: weight on violation correction at first touch
    solver_impedance_min: float = 0.98  # MuJoCo default: 0.9
    # Max impedance: weight on violation correction when penetration is very deep
    solver_impedance_max: float = 0.99  # MuJoCo default: 0.95
    # Width: over what penetration distance does the impedance go from min to max
    solver_impedance_min2max_width: float = 1e-5  # MuJoCo default: 1e-3
    # Midpoint: where in the min-to-max width "mid" force is applied (leave as is)
    solver_impedance_transitionmidpoint: float = 0.5  # MuJoCo default: 0.5
    # Sharpness: how sharp the transition from min to max is (1 = linear)
    solver_impedance_transitionsharpness: float = 3.0  # MuJoCo default: 2.0

    def get_friction_tuple(self):
        """Get friction coefficients expected by MuJoCo. Note that MuJoCo uses
        all 5 values for contact pairs. This is different from the friction
        attribute of geometries (only a 3-tuple)."""
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
        """Get solver reference acceleration parameters expected by MuJoCo."""
        self._raise_on_invalid_solver_refaccl()
        return (
            self.solver_refaccl_timeconst,
            self.solver_refaccl_dampratio,
        )

    def get_solimp_tuple(self):
        """Get solver impedance parameters expected by MuJoCo."""
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
                raise e
            print(f"Invalid contact parameters: {e}.")
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
