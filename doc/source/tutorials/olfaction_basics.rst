Olfaction basics
================

**Author:** Sibo Wang-Chen

**Note:** The code presented in this notebook has been simplified and
restructured for display in a notebook format. A more complete and
better structured implementation can be found in the `examples folder of
the FlyGym repository on
GitHub <https://github.com/NeLy-EPFL/flygym/tree/main/flygym/examples/>`__.

**Summary:** In this tutorial, we will implement a simple controller for
odor-guided taxis.

In the `previous
tutorial <https://neuromechfly.org/tutorials/vision.html>`__, we covered
how one can simulate the visual experience of the fly simulation. In
addition to vision, we also made it possible for our model to detect
odors emitted by objects in the simulation environment. The olfactory
system in *Drosophila* consists of specialized olfactory sensory neurons
(OSNs) located in the antennae and maxillary palps. These detect
specific odorant molecules and convey this information to the brain’s
antennal lobe, where their signals are further processed. This is shown
in the figure below (left, source: `Martin et al,
2013 <https://doi.org/10.1002/ar.22747>`__) We emulated peripheral
olfaction by attaching virtual odor sensors to the antennae and
maxillary palps of our biomechanical model, as shown in the figure
(right). The user has the option of configuring additional sensors at
more precise locations on these olfactory organs. These virtual sensors
can detect odor intensities across a multi-dimensional space that can be
thought of as representing, for example, the concentrations of
monomolecular chemicals sensed by OSNs in the antennae, or the
intensities of composite odors co-activating numerous projection neurons
in the antennal lobe.

.. image:: https://github.com/NeLy-EPFL/_media/blob/main/flygym/olfaction_basics/olfaction.png?raw=true


Odor arena
----------

To demonstrate odor sensing, let’s create an environment with one
attractive odor source and two aversive odor sources. The dimension of
this odor space is 2 (attractive, aversive) despite the number of odor
sources being 3. The odor sources share a peak intensity of 1. We will
color the attractive odor source orange and the aversive odor sources
blue.

.. code:: ipython3

    import numpy as np
    
    # Odor source: array of shape (num_odor_sources, 3) - xyz coords of odor sources
    odor_source = np.array([[24, 0, 1.5], [8, -4, 1.5], [16, 4, 1.5]])
    
    # Peak intensities: array of shape (num_odor_sources, odor_dimensions)
    # For each odor source, if the intensity is (x, 0) then the odor is in the 1st dimension
    # (in this case attractive). If it's (0, x) then it's in the 2nd dimension (in this case
    # aversive)
    peak_odor_intensity = np.array([[1, 0], [0, 1], [0, 1]])
    
    # Marker colors: array of shape (num_odor_sources, 4) - RGBA values for each marker,
    # normalized to [0, 1]
    marker_colors = [[255, 127, 14], [31, 119, 180], [31, 119, 180]]
    marker_colors = np.array([[*np.array(color) / 255, 1] for color in marker_colors])
    
    odor_dimensions = len(peak_odor_intensity[0])

Let’s create the arena using these parameters. The detailed
documentation of the ``OdorArena`` class can be found in the `API
reference <https://neuromechfly.org/api_ref/arena.html#flygym.arena.OdorArena>`__.
Its implementation is beyond the scope of this tutorial but can be found
`here <https://github.com/NeLy-EPFL/flygym/blob/main/flygym/arena/sensory_environment.py>`__.

.. code:: ipython3

    from flygym.arena import OdorArena
    
    arena = OdorArena(
        odor_source=odor_source,
        peak_odor_intensity=peak_odor_intensity,
        diffuse_func=lambda x: x**-2,
        marker_colors=marker_colors,
        marker_size=0.3,
    )

Let’s place our fly in the arena. As before, we will run a few
iterations to allow it to stand on the ground in a stable manner.

.. code:: ipython3

    import matplotlib.pyplot as plt
    from flygym import Fly, Camera
    from flygym.examples.locomotion import HybridTurningController
    from pathlib import Path

    outputs_dir = Path("./outputs/olfaction_basics")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    
    fly = Fly(
        spawn_pos=(0, 0, 0.2),
        contact_sensor_placements=contact_sensor_placements,
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=False,
    )
    
    cam = Camera(
        fly=fly,
        camera_id="birdeye_cam",
        play_speed=0.5,
        window_size=(800, 608),
    )
    
    sim = HybridTurningController(
        fly=fly,
        cameras=[cam],
        arena=arena,
        timestep=1e-4,
    )
    for i in range(500):
        sim.step(np.zeros(2))
        sim.render()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
    ax.imshow(cam._frames[-1])
    ax.axis("off")
    fig.savefig(outputs_dir / "olfaction_env.png")



.. image:: https://github.com/NeLy-EPFL/_media/blob/main/flygym/olfaction_basics/olfaction_env.png?raw=true


Controller for odor taxis
-------------------------

Let’s design a simple hand-tuned controller for odor-guided taxis. We
start by calculating the left-right asymmetry of the odor intensity
:math:`I` for each odor :math:`o`:

.. math::


   \Delta I_o = \frac{I_\text{left,o} - I_\text{right,o}}{(I_\text{left,o} + I_\text{right,o}) / 2}

Then, we multiply :math:`\Delta I_o` by a gain :math:`\gamma_o` for each
odor dimension and take the sum :math:`s`. Attractive and aversive odors
will have different signs in their gains.

.. math::


   s = \sum_{o} \gamma_o \Delta I_o

We transform :math:`s` nonlinearly to avoid overly drastic turns when
the asymmetry is subtle and to crop it within the range [0, 1). This
gives us a turning bias :math:`b`:

.. math::


   b = \tanh(s^2)

Finally, we modulate the descending signal :math:`\delta` based on
:math:`b` and the sign of :math:`s`:

.. math::


   \delta_\text{left} = 
       \begin{cases}
       \delta_\text{max} & \text{if } s>0\\
       \delta_\text{max} - b(\delta_\text{max} - \delta_\text{min})  & \text{otherwise}
       \end{cases}
       \qquad
       \delta_\text{right} = 
       \begin{cases}
       \delta_\text{max} - b(\delta_\text{max} - \delta_\text{min}) & \text{if } s>0\\
       \delta_\text{max}  & \text{otherwise}
       \end{cases}

where, :math:`\delta_\text{min}`, :math:`\delta_\text{max}` define the
range of the descending signal. Here, we will use the following
parameters:

-  :math:`\gamma_\text{attractive} = -500` (negative ipsilateral gain
   leads to positive taxis)
-  :math:`\gamma_\text{aversive} = 80` (positive ipsilateral gain leads
   to negative taxis)
-  :math:`\delta_\text{min} = 0.2`
-  :math:`\delta_\text{max} = 1`

As before, we will recalculate the steering signal every 0.05 seconds.
Let’s implement this in Python:

.. code:: ipython3

    from tqdm import trange
    
    attractive_gain = -500
    aversive_gain = 80
    decision_interval = 0.05
    run_time = 5
    num_decision_steps = int(run_time / decision_interval)
    physics_steps_per_decision_step = int(decision_interval / sim.timestep)
    
    obs_hist = []
    odor_history = []
    obs, _ = sim.reset()
    for i in trange(num_decision_steps):
        attractive_intensities = np.average(
            obs["odor_intensity"][0, :].reshape(2, 2), axis=0, weights=[9, 1]
        )
        aversive_intensities = np.average(
            obs["odor_intensity"][1, :].reshape(2, 2), axis=0, weights=[10, 0]
        )
        attractive_bias = (
            attractive_gain
            * (attractive_intensities[0] - attractive_intensities[1])
            / attractive_intensities.mean()
        )
        aversive_bias = (
            aversive_gain
            * (aversive_intensities[0] - aversive_intensities[1])
            / aversive_intensities.mean()
        )
        effective_bias = aversive_bias + attractive_bias
        effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
        assert np.sign(effective_bias_norm) == np.sign(effective_bias)
    
        control_signal = np.ones((2,))
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * 0.8
        control_signal[side_to_modulate] -= modulation_amount
    
        for j in range(physics_steps_per_decision_step):
            obs, _, _, _, _ = sim.step(control_signal)
            rendered_img = sim.render()
            if rendered_img is not None:
                # record odor intensity too for video
                odor_history.append(obs["odor_intensity"])
            obs_hist.append(obs)
    
        # Stop when the fly is within 2mm of the attractive odor source
        if np.linalg.norm(obs["fly"][0, :2] - odor_source[0, :2]) < 2:
            break


.. parsed-literal::

     51%|█████     | 51/100 [00:57<00:55,  1.13s/it]


We can visualize the fly trajectory:

.. code:: ipython3

    fly_pos_hist = np.array([obs["fly"][0, :2] for obs in obs_hist])
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), tight_layout=True)
    ax.scatter(
        [odor_source[0, 0]],
        [odor_source[0, 1]],
        marker="o",
        color="tab:orange",
        s=50,
        label="Attractive",
    )
    ax.scatter(
        [odor_source[1, 0]],
        [odor_source[1, 1]],
        marker="o",
        color="tab:blue",
        s=50,
        label="Aversive",
    )
    ax.scatter([odor_source[2, 0]], [odor_source[2, 1]], marker="o", color="tab:blue", s=50)
    ax.plot(fly_pos_hist[:, 0], fly_pos_hist[:, 1], color="k", label="Fly trajectory")
    ax.set_aspect("equal")
    ax.set_xlim(-1, 25)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.6))
    fig.savefig(outputs_dir / "odor_taxis_trajectory.png")



.. image:: https://github.com/NeLy-EPFL/_media/blob/main/flygym/olfaction_basics/odor_taxis_trajectory.png?raw=true


We can also generate the video:

.. code:: ipython3

    cam.save_video(outputs_dir / "odor_taxis.mp4")


.. raw:: html

   <video src="https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/olfaction_basics/odor_taxis.mp4" controls="controls" style="max-width: 400px;"></video>
