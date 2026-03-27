from pathlib import Path
from importlib.resources import files

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from flygym.anatomy import JointDOF


class MotionSnippet:
    def __init__(
        self,
        data_path: Path | None = None,
        *,
        angles_global2anatomical: bool = True,
    ) -> None:
        if data_path is None:
            module_dir = Path(str(files("flygym_examples.spotlight_data")))
            data_path = module_dir / "assets/spotlight_behavior_clip.npz"
        data = np.load(data_path, allow_pickle=True)
        self.rawpred_egoxyz = data["rawpred_egoxyz"]
        self.fwdkin_egoxyz = data["fwdkin_egoxyz"]
        self.joint_angles = data["joint_angles"].copy()  # (nsteps, 6 legs, 7 dofs/leg)
        self.keypoints = [tuple(x) for x in data["keypoints"].tolist()]
        self.legs = data["legs"].tolist()
        self.dofs_per_leg = [tuple(x) for x in data["dofs_per_leg"].tolist()]
        self.experiment_trial = data["experiment_trial"].item()
        self.framerange_in_raw_recording = data["framerange_in_raw_recording"].tolist()
        self.data_fps = data["data_fps"].item()

        if angles_global2anatomical:
            self._apply_global2anatomical()

    def _apply_global2anatomical(self) -> None:
        """Flip the sign of right-leg roll and yaw angles.

        SeqIKPy (the upstream IK library) defines joint angles in a global
        frame, so "outward" rotations on left vs. right legs carry opposite
        signs.  NeuroMechFly uses an anatomical convention where left/right
        are symmetric, so we flip the sign here once at load time rather than
        at every call site.
        """
        right_leg_indices = [i for i, leg in enumerate(self.legs) if leg[0] == "r"]
        mirror_dof_indices = [
            i
            for i, (_, _, axis) in enumerate(self.dofs_per_leg)
            if axis in ("roll", "yaw")
        ]
        for leg_idx in right_leg_indices:
            for dof_idx in mirror_dof_indices:
                self.joint_angles[:, leg_idx, dof_idx] *= -1

    def get_joint_angles(
        self,
        output_timestep: float,
        output_dof_order: list[JointDOF],
        *,
        sgfilter_window_sec: float = 0.03,
        sgfilter_polyorder: int = 3,
    ) -> np.ndarray:
        """Return smoothed, interpolated joint angles reordered to match
        ``output_dof_order``.

        Parameters
        ----------
        output_timestep:
            Simulation timestep in seconds (e.g. 1e-4).
        output_dof_order:
            List of JointDOF objects in the order expected by the simulator
            (i.e. from ``fly.get_actuated_jointdofs_order(actuator_type)``).
        sgfilter_window_sec:
            Savitzky-Golay filter window length in seconds.
        sgfilter_polyorder:
            Polynomial order for the Savitzky-Golay filter (also used as the
            spline degree for subsequent cubic interpolation).

        Returns
        -------
        np.ndarray, shape (n_output_steps, len(output_dof_order))
        """
        # --- 1. Savitzky-Golay smoothing ---
        window_size = int(sgfilter_window_sec * self.data_fps)
        window_size += 1 - (window_size % 2)  # must be odd
        angles_filtered = savgol_filter(
            self.joint_angles,
            window_length=window_size,
            polyorder=sgfilter_polyorder,
            axis=0,
        )

        # --- 2. Cubic interpolation onto the simulation time grid ---
        n_frames = self.joint_angles.shape[0]
        duration_sec = n_frames / self.data_fps
        source_timegrid = np.arange(n_frames) / self.data_fps
        output_timegrid = np.arange(0, duration_sec, output_timestep)

        f = interp1d(
            source_timegrid,
            angles_filtered,
            kind="cubic",
            axis=0,
            bounds_error=False,
            fill_value=(angles_filtered[0], angles_filtered[-1]),
        )
        angles_interp = f(output_timegrid)

        # --- 3. Reorder axes to match the simulator's expected DOF order ---
        leg_dof_pairs = np.array(
            [
                (
                    self.legs.index(dof.child.pos),
                    self.dofs_per_leg.index(
                        (dof.parent.link, dof.child.link, dof.axis.value)
                    ),
                )
                for dof in output_dof_order
            ],
            dtype=np.int32,
        )  # shape: (n_dofs, 2)

        return angles_interp[:, leg_dof_pairs[:, 0], leg_dof_pairs[:, 1]]
