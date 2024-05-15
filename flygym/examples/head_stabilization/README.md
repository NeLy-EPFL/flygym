## FlyGym examples: Head stabilization

> [!NOTE]
> `flygym/examples` aims to provide a packaged and importable implementation of things explained in the [online tutorials](https://neuromechfly.org/tutorials/index.html). The tutorials offer a much more detailed walk-through of the concepts and it is preferable to start from there.

This subpackage includes the following files:
- `collect_training_data.py`: This script simulates fly walking over simple and complex terrains with different gaits.
- `check_videos.py`: We used this script to manually exclude simulations where the fly flipped while walking over complex terrain. To use it,
  - First run it to plot the last frame of all simulations in one image (`last_frames_all.png`).
  - Modify the `excluded_videos` variable to indicate the trials to be excluded.
  - Run it again and check `last_frames_clean.png`. This time, no panel should contain a flipped fly.
- `model.py`: This module defines the following —
  - A PyTorch dataset that draws data from the training trials,
  - The artificial neural network model used to compute the appropriate neck actuation signals,
  - A data scaler to be used with the model, and
  - An inference-time wrapper of the model suitable for closed-loop deployment.
- `train_proprioception_model.py`: This script trains and evaluates models defined above using the simulated trials.
- `closed_loop_deployment.py`: This script simulates fly walking over simple and complex terrain, with and without the trained head stabilization model, to demonstrate the model's efficacy.
- `util.py`: Utility for data serialization.
- `viz.py`: Visualization functions.