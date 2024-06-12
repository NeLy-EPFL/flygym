rm -rf outputs/
mkdir -p outputs
mkdir -p executed_notebooks


# Function to run a notebook and save the executed version
run_notebook() {
    file=$1
    printf "********** Executing $file ($(date)) **********\n"
    jupyter nbconvert --to notebook --execute $file --output="executed_notebooks/$file"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Error executing $file"
        exit 1
    fi
    printf "Done executing $file ($(date))\n\n"
}


# Gym basics
run_notebook "gym_basics_and_kinematic_replay.ipynb"

# CPG-based controller
run_notebook "cpg_controller.ipynb"

# Rule-based controller
run_notebook "rule_based_controller.ipynb"

# Hybrid controller
run_notebook "hybrid_controller.ipynb"

# Turning controller
run_notebook "turning.ipynb"

# Vision basics
run_notebook "vision_basics.ipynb"

# Olfaction basics
run_notebook "olfaction_basics.ipynb"

# Path integration
run_notebook "path_integration.ipynb"
rm outputs/path_integration/sim_data.pkl
# TODO: remove downloaded files

# Head stabilization
run_notebook "head_stabilization.ipynb"
rm -rf outputs/head_stabilization/logs/
rm -rf outputs/head_stabilization/models/
rm -rf outputs/head_stabilization/tripod_blocks_train_set_1.00_1.00/
rm -rf outputs/head_stabilization/tripod_flat_train_set_1.00_1.00
rm -rf outputs/head_stabilization/head_stabilization_joint_angle_scaler_params.pkl
# TODO: remove downloaded files

# Advanced vision
run_notebook "advanced_vision.ipynb"

# Advanced olfaction
run_notebook "advanced_olfaction.ipynb"
rm outputs/plume_tracking/plume.hdf5
rm outputs/plume_tracking/plume_tcropped.hdf5
