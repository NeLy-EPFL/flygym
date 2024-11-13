Camera
======

The ``Camera`` class defines how images should be rendered from a camera in the world model. Note that the ``Camera`` class does not by itself add a camera to the MuJoCo model — the camera must have already existed and can be referenced by its name. The camera can be added in two ways:

1. By adding a camera to the MuJoCo model file. Refer to the `section on camera <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera>`_ in the MuJoCo XML reference for more information. As an example, the following XML code adds a tracking camera to the MuJoCo model:

.. code-block:: xml

    <worldbody>
        <!-- ... other things ... -->
        <camera name="camera_top" class="nmf" mode="track" ipd="0.068" pos="0 0 8" euler="0 0 0"/>
    </worldbody>

2. By calling ``.worldbody.add()`` on the root element of the MuJoCo model programmatically in Python. Practically, this can be done by extending an existing FlyGym Arena class and adding the camera in the ``__init__`` method. For example, the following code adds a stationary bird's eye camera to the MuJoCo model:

.. code-block:: python3

    from flygym.arena import BaseArena

    class MyArena(BaseArena):
        def __init__(self):
            super().__init__()
            self.birdeye_cam = self.root_element.worldbody.add(
                "camera",
                name="birdseye_cam",
                mode="fixed",
                pos=(20, 0, 20),
                euler=(0, 0, 0),
                fovy=60,
            )
            # ... other things ...

Once the camera is added to the MuJoCo model, it can be referenced by its name to initialize the ``Camera`` object with additional parameters assigned to it. These can include: rendering rate in frames-per-second (FPS), window size, play speed, etc. Furthermore, the ``Camera`` class has useful methods such as ``.save_video``, ``.reset``, etc. Logics such as rotating the camera in tilted terrain are also implemented in the ``Camera`` class. The ``.render`` method is the main point of entry for fetching images from the camera.

The full API reference of the ``Camera`` class is as follows:

.. autoclass:: flygym.camera.Camera
    :members:
    :undoc-members:
    :show-inheritance: