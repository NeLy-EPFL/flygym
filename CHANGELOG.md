# Changelog

## Version 2.1.0

> [!CAUTION]
> ### API-breaking changes (PyMJCF → MjSpec migration)
> FlyGym 2.0.3 drops the `dm-control` PyMJCF backend in favour of MuJoCo's native `MjSpec` API. Model elements (cameras, materials, bodies, joints, …) are now `mujoco._specs.Mjs*` objects rather than `dm_control.mjcf` wrappers, and two patterns that worked under PyMJCF need to be updated:
>
> - **`element.full_identifier` → `element.name`:** PyMJCF exposed `full_identifier` to return the fully-scoped compiled name (e.g. `"fly/trackcam"`). MjSpec mutates `.name` in place when a child spec is attached with a prefix, so `.name` already returns the prefixed name after `world.add_fly()`. Replace every occurrence of `.full_identifier` with `.name`.
> - **`spec.asset.find_all("material")` → `spec.materials`:** The PyMJCF tree-traversal helper is gone. MjSpec exposes typed collection properties (`spec.materials`, `spec.bodies`, `spec.joints`, etc.) directly on the `MjSpec` object.

> [!NOTE]
> ### Migration guide: `element.full_identifier`
> If you stored references to spec elements (e.g. the `MjsCamera` returned by `fly.add_tracking_camera()`) and later used `element.full_identifier` as a dict key or to look up the compiled ID, replace it with `element.name`:
>
> ```python
> # Before (PyMJCF)
> cam = fly.add_tracking_camera(name="body_cam")
> world.add_fly(fly, ...)
> frames = renderer.frames[cam.full_identifier]  # AttributeError under MjSpec
>
> # After (MjSpec)
> cam = fly.add_tracking_camera(name="body_cam")
> world.add_fly(fly, ...)
> frames = renderer.frames[cam.name]  # "fly_name/body_cam" after attachment
> ```
>
> No other change is needed: `cam.name` already returns the same prefixed string that `cam.full_identifier` used to return, because MjSpec mutates held element references in place when the child spec is attached.

> [!CAUTION]
> ### `add_tracking_camera` placement is now relative to the fly's root segment
> `Fly.add_tracking_camera()` now adds the camera *inside* the fly's root segment (thorax) body so that MuJoCo's `track` mode actually follows the fly as it moves. As a result, `pos_offset` (and `rotation`) are interpreted in the root segment's body frame rather than in world coordinates as before, and the default `pos_offset` changed from `(0, -7.5, 6)` to `(-0.5, -7.5, 5)`.
>
> **Any hard-coded `pos_offset` values must be re-tuned.** A value that previously positioned the camera in the world frame now positions it relative to the root segment, which in the neutral pose sits roughly `(0.5, 0, 1.3)` mm from the fly's attachment point plus the spawn height. The camera will therefore appear higher and shifted toward the head unless you adjust the offset (to reproduce the old framing, subtract the root segment's rest position from your previous world-frame offset).
>
> The same `pos_offset` now yields the same camera position *relative to the fly* in every context (`FlatGroundWorld`, `TetheredWorld`, or a fly compiled on its own for `preview_model`).

### Bug fixes
- Fixed `AttributeError: 'mujoco._specs.MjsCamera' object has no attribute 'full_identifier'` in tutorials 4a–4d and 5b caused by the PyMJCF → MjSpec migration ([#285](https://github.com/NeLy-EPFL/flygym/pull/285)).

## Version 2.0.2
> [!CAUTION]
> ### API-breaking changes
> - **Adhesion control range**: `Fly.add_leg_adhesion()` and `Simulation.set_leg_adhesion_states()` now use a normalised `[0, 1]` control range (0 = fully released, 1 = > full gain) instead of the previous `[1, 100]` range. Boolean `0`/`1` adhesion commands that previously fell outside the valid range will now work as expected.
> - **`get_ground_contact_info()` first return value**: renamed from `contact_active` to `contact_found` and now returns the raw MuJoCo contact-sensor `found` channel (a > floating-point value) rather than a booleanized flag. Code that compared `contact_active == 1` should now check `contact_found > 0`.
> - **`ContactParams.get_solimp_tuple()`**: now returns a 5-element tuple (was 4). The previously missing `solver_impedance_min2max_width` field is now included as the > third element (`[dmin, dmax, width, midpoint, power]`), matching MuJoCo's `solimp` specification. Any code that destructured or indexed the old 4-tuple needs to be updated.

### Bug fixes
- `ContactParams.get_solimp_tuple()` previously returned a 4-element tuple, causing MuJoCo to silently use a default value for the transition-width parameter. It now returns the correct 5-element tuple.

### Additions
- Added vision to CPU-based simulation. Addition of vision to GPU-based simulation will be deferred for now.
- Added `GappedTerrainWorld`, `BlocksTerrainWorld`, and `MixedTerrainWorld` to `flygym.compose`.
- Added `Simulation.get_bodysegment_contact_forces()` for querying ground-contact forces on arbitrary body segments.
- Added locomotion controller examples (`CPGController`, `RuleBasedController`, `HybridController`, `HybridTurningController`) in `flygym_demo.complex_terrain`. Added tutorials accordingly.

### Housekeeping
- Switched formatter from Black to Ruff


## Version 2.0.1 (2026–04–16)
### Bug fixes
- Include sample Spotlight data in shipped `flygym_demo` build ([#259](https://github.com/NeLy-EPFL/flygym/pull/259)).
- Fix dimension check for 3D rotations specified in `axisangle` format.
- Fix bugs with creating `AnatomicalJoint` and `JointDOF` objects from strings.

### Additions
- New API to add sites to fly model and read out their positions
- Additional tutorial: _1bis. Advanced model composition_

### Housekeeping
- Add changelog and correct info in LICENSE

### Contributors
@sibocw, @djsamseng

## Version 2.0.0 (2026–04–02)
> [!CAUTION]
> This version is a full rewrite of the codebase and it is not backward-compatible. See [this page](https://github.com/NeLy-EPFL/flygym/blob/v2.0.0/docs/migration.md) for more info.


### Version 1.2.1

* Improve vision readout computing time by avoiding redundant memory copies
* Bump NumPy version to 2.* and end official support for macOS 13

## Version 1.2.0

> [!CAUTION]
> ### API-breaking changes
> * Enhance camera logic. See the [Camera API reference](api_ref/camera.html) for details.
> * Use the [FlyVis package](https://github.com/TuragaLab/flyvis) as [published on PyPI](https://pypi.org/project/flyvis/). Users should now use ``flyvis`` instead of ``flyvision`` as module name.

### Other changes
* Add NeuroMechFly game for outreach (used at EPFL Scientastic! Days, etc.).
* Transition from ``setup.py`` to ``pyproject.toml``, specifically using [Poetry](https://python-poetry.org/) as the build backend. Users can now install FlyGym with the exact versions of dependencies used by the developers by running ``poetry install`` from the root directory. This will create a virtual environment with the correct dependencies as specified in the included ``poetry.lock`` file, which is version-tracked as a part of the FlyGym Github repository.
* Add [VS Code devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) support.
* Remove support for Python 3.9.
   
## Version 1.1.0
* Added cardinal direction sensing (vectors describing +x, +y, +z of the fly) to the observation space.
* Removed legacy spawn orientation preprocessing: Previously, pi/2 was subtracted from the user-specified spawn orientation on the x-y plane. This was to make the behavior consistent with a legacy version of NeuroMechFly. This behavior is no longer desired; from this version onwards, the spawn orientation is used as is.
* Strictly fixed the required MuJoCo version to 3.2.3, and dm_control version to 1.0.23. This is to prevent API-breaking changes in future versions of these libraries from affecting FlyGym. FlyGym maintainers will periodically check for compatibility with newer versions of these libraries.
* Changed flip detection method: Previously, flips are reported when all legs reliably lose contact with the ground. Now, we simply check if the z component of the "up" cardinal vector is negative. Additionally, the ``detect_flip`` parameter of ``Fly`` is now deprecated; flips are always detect and reported.
* Allowed different sets of DoFs to be monitored vs. actuated. Previously, the two sets are always the same.
* From this version onwards, we will use [EffVer](https://jacobtomlinson.dev/effver/) as the versioning policy. The version number will communicate how much effort we expect a user will need to spend to adopt the new version. While we previously tried to adhere to the stricter [SemVer](https://semver.org/), we found that it was not effective because many core dependencies of FlyGym (e.g., MuJoCo, NumPy, and Python itself) do not use SemVer.

## Version 1.0.1
Fixed minor bugs related to the set of DoFs in the predefined poses, and to rendering at extremely high frequencies. Fixed outdated class names and links in the docs. In addition, contact sensor placements used by the hybrid turning controller are now added to the ``preprogrammed`` module.

## Version 1.0.0
In spring 2024, NeuroMechFly was used, for the second time, in a course titled [Controlling behavior in animals and robots](https://edu.epfl.ch/coursebook/en/controlling-behavior-in-animals-and-robots-BIOENG-456) at EPFL. At the same time, we revised the NeuroMechFly v2 manuscript. In these processes, we significantly improved the FlyGym package, added new functionalities, and incorporated changes as we received feedback from the students. These enhancements are released as FlyGym version 1.0.0. This release is not backward compatible; please refer to the [tutorials](https://neuromechfly.org/tutorials/index.html) and [API references](https://neuromechfly.org/api_ref/index.html) for more information. The main changes are:

> [!IMPORTANT]
> ### Major API changes:
> * The ``NeuroMechFly`` class is split into ``Fly``, a class that represents the fly, and ``Simulation``, a class that represents the simulation, which can potentially contain multiple flies.
> * The ``Parameters`` class is deprecated. Parameters related to the fly (such as joint parameters, actuated DoFs, etc.) should be set directly on the ``Fly`` object. Parameters related to the simulation (such as the time step, the render cameras, etc.) should be set directly on the ``Simulation`` object.
> * A new ``Camera`` class is introduced. A simulation can contain multiple cameras.

### New examples
See [examples](https://github.com/NeLy-EPFL/flygym/tree/main/flygym/examples):

* Path integration based on ascending mechanosensory feedback.
* Head stabilization based on ascending mechanosensory feedback.
* Navigating a complex plume, simulated separately in a fluid mechanics simulator.
* Following another fly using a realistic, connectome-constrained neural network that processes visual inputs.

## Version 0.2.5
Modify model file to make it compatible with MuJoCo 3.1.1. Disable Python 3.7 support accordingly.

## Version 0.2.4
Set MuJoCo version to 2.3.7. Documentation updates.

## Version 0.2.3
Various bug fixes. Improved placement of the spherical treadmill in the tethered environment.

## Version 0.2.2
Changed default joint kp and adhesion forces to those used in the controller comparison task. Various minor bug fixes. Documentation updates.

## Version 0.2.1
Simplified class names: ``NeuroMechFlyMuJoCo`` → ``NeuroMechFly``, ``MuJoCoParameters`` → ``Parameters``. Minor documentation updates.

## Version 0.2.0
The current base version — major API change from 0.1.x.

## Version 0.1.x
Versions used during the initial development of NeuroMechFly v2.

## Unversioned
Versions used for the Spring 2023 offering of BIOENG-456 Controlling Behavior in Animals and Robots course at EPFL.
