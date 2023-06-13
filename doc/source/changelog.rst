Change Log
==========

* **2023-06-10:** A series of important API changes:

  * "Terrain" is now renamed "Arena" to better generalize to environments with higher-order features. The user must now define an arena object and pass it to ``flygym.envs.nmf_mujoco.NeuroMechFlyMuJoCo.__init__`` (whereas previously it was the name of the terrain type as a string). ``terrain_config`` is removed as an argument; the arena should be configured in its own ``__init__`` method. The exception is the parameters controlling how the fly is spawned in the arena: ``NeuroMechFlyMuJoCo.__init__`` now accepts the ``spawn_pos`` and ``spawn_orient`` arguments.
  * Several physics simulation parameters and rendering parameters are subsumed into ``flygym.envs.nmf_mujoco.MuJoCoParameters``. The following arguments to ``flygym.envs.nmf_mujoco.NeuroMechFlyMuJoCo.__init__`` are removed: ``timestep``, ``render_mode``, ``physics_config`` and ``render_config``. 
  * ``flygym.envs.nmf_mujoco.MuJoCoParameters.step()`` now returns the observation, reward, terminated, truncated, info to fully comply with the Gym Env API.
  * The following attributes of ``flygym.envs.nmf_mujoco.NeuroMechFlyMuJoCo`` have been renamed:
    
    * ``self_contact_pairs`` → ``self_contacts``
    * ``self_contact_pairs_names`` → ``self_contact_names``
    * ``floor_contact_pairs`` → ``floor_contacts``
    * ``floor_contact_pairs_names`` → ``floor_contact_names``
    * ``collision_tracked_geoms`` → ``contact_sensor_placements``

* **2023-04-06:** In the MuJoCo environment, ``.reset()`` will now reset the fly to its initial pose.

* **2023-04-06:** In the MuJoCo environment, ``.save_video(path: pathlib.Path)`` is now available to explicitly save the rendered video. This is useful when the user wishes to run some simulation, save the video, reset the environment, and run more simulation using the same environment.

* **2023-04-06:** In the MuJoCo environment, if ``data_dir`` is not provided upon initialization, a new directory will no longer be created. In that, no data will be saved upong calling ``.close()`` to close the environment. The user can use ``.save_video(path: pathlib.Path)`` to explicitly save the video.
