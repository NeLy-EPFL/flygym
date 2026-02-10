# APPLICABLE TO MACOS USERS ONLY
# If using uv for package management, first link the Python dynamic library
# (libpython3.*.dylib) to the virtual environment folder. This is because uv does not
# ordinarily link the Python dynamic library, but that dylib is required for the
# mujoco-specific python wrapper mjpython. See details at
# https://github.com/google-deepmind/mujoco/issues/1923#issuecomment-2752784740

source .venv/bin/activate

# Find the path to LIBDIR and LDLIBRARY (dylib file) on your system:
PYTHON_LIB_DIR=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
PYTHON_LIB_NAME=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')

# Soft link it inside your virtual environment:
ln -s "$PYTHON_LIB_DIR/$PYTHON_LIB_NAME" "./.venv/$PYTHON_LIB_NAME"

if [ -L "./.venv/$PYTHON_LIB_NAME" ] && [ -e "./.venv/$PYTHON_LIB_NAME" ]; then
    echo "Successfully linked $PYTHON_LIB_NAME to .venv directory."
else
    echo "Failed to link $PYTHON_LIB_NAME. Please check the paths and try again."
fi