# https://github.com/google-deepmind/mujoco/issues/1923#issuecomment-2752784740

source .venv/bin/activate

# Find the path to LIBDIR and LDLIBRARY (dylib file) on your system:
PYTHON_LIB_DIR=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
PYTHON_LIB_NAME=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')

# Soft link it inside your virtual environment:
ln -s "$PYTHON_LIB_DIR/$PYTHON_LIB_NAME" "./.venv/$PYTHON_LIB_NAME"
