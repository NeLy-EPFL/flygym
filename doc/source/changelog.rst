Change Log
==========

* **2023-04-06:** In the MuJoCo environment, ``.reset()`` will now reset the fly to its initial pose.
* **2023-04-06:** In the MuJoCo environment, ``.save_video(path: pathlib.Path)`` is now available to explicitly save the rendered video. This is useful when the user wishes to run some simulation, save the video, reset the environment, and run more simulation using the same environment.
* **2023-04-06:** In the MuJoCo environment, if ``data_dir`` is not provided upon initialization, a new directory will no longer be created. In that, no data will be saved upong calling ``.close()`` to close the environment. The user can use ``.save_video(path: pathlib.Path)`` to explicitly save the video.
