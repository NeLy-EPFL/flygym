Camera
======

The ``Camera`` class defines how images should be rendered from a camera in the world model.
Cameras are added dynamically to the model by calling ``.body.add()`` on any body of the MuJoCo model programmatically in Python. In  this way, the XML file does not need to contain all the preset cameras. 
When instantiating a ``Camera`` object, the user has to specify:
 - The attachement point this can be a body or a site in the model.
 - The name of the camera.
 - The targeted fly names. Those flies will be the one(s) whose contact forces will be drawn.
In order to simplify ``Camera`` instantiation, we provide a set of predefined camera parameters in config.yaml. Those parameters are used if the camera name matches the name of one of those cameras.
In case the camera name does not match any of the predefined cameras, the user can specify the camera parameters manually.

This new logic for cameras allows a more flexible definition of the cameras update rules.
We propose three different camera update rules:
- ``ZStabilizedCamera``: The camera z position is fixed at a given height above the floor.
- ``YawOnlyCamera``: Only the yaw and position of the camera are updated this smoothen camera movements during locomotion in tracked cameras.
- ``GravityAlignedCamera``: This camera updates its orientation based on the changes in the gravity vector. This camera is useful for tracking the fly orientation in the world frame.

The full API reference of the different type of camera classes is as follows:

.. automodule:: flygym.camera
    :members:
    :undoc-members:
    :show-inheritance: