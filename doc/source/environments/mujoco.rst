MuJoCo Specifics
================

Example
-------

The following code snippet executes an environment where all leg joints of the fly repeat a sinusoidal motion. The output will be saved as a video and the observation will be appended to a list. ::

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo

    # First, we initialize simulation
    run_time = 0.1
    out_dir = Path('mujoco_basic_untethered_sinewave')
    nmf = NeuroMechFlyMuJoCo(render_mode='saved', output_dir=out_dir)

    # Define the frequency, phase, and amplitude of the sinusoidal waves
    freq = 500
    phase = 2 * np.pi * np.random.rand(len(nmf.actuators))
    amp = 0.9

    obs_list = []    # keep track of the observed states
    while nmf.curr_time <= run_time:    # main loop
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {'joints': joint_pos}
        obs, info = nmf.step(action)
        nmf.render()
        obs_list.append(obs)
    nmf.close()


API Reference
-------------

We provide a comprehensive API reference to the MuJoCo environment below.

.. autoclass:: flygym.envs.nmf_mujoco.NeuroMechFlyMuJoCo
   :members: __init__, reset, step, render, close