# ----------------------------------------------------------------------
# Usage:
#   source rock_setup.sh
#
# UV runtime environment setup:
#   1. Sets ROCM_PATH and HIP_PATH
#   2. Adds ROCm runtime libraries to LD_LIBRARY_PATH and LIBRARY_PATH
#   3. Adds ROCm device libraries (ROCM_DEVICE_LIB_PATH / HIP_DEVICE_LIB_PATH)
#   4. Adds ROCm headers and thrust include paths (CPLUS_INCLUDE_PATH, CPATH)
# ----------------------------------------------------------------------

# Safety check: must be sourced, not executed
if [ "$0" = "$BASH_SOURCE" ]; then
  echo "[WARN] Please run this script with: source $0"
  exit 1
fi

#Check for Conda environment
if [ -n "$CONDA_PREFIX" ]; then
  echo "[INFO] Conda environment found: $CONDA_PREFIX"
  _ENV_PREFIX="$CONDA_PREFIX"
elif [ -n "$VIRTUAL_ENV" ]; then
  echo "[INFO] Python venv found: $VIRTUAL_ENV"
  _ENV_PREFIX="$VIRTUAL_ENV"
else
  echo "[ERROR] No environment found"
  _ENV_PREFIX=$(python -c "import sys; print(sys.prefix)")
fi
_PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO] Python version: $_PYVER"
echo "[INFO] Environment prefix: $_ENV_PREFIX"

#Set the environment variables for the ROCm SDK

export ROCM_PATH=$_ENV_PREFIX/lib/python$_PYVER/site-packages/_rocm_sdk_core
export ROCM_DEV_PATH=$_ENV_PREFIX/lib/python$_PYVER/site-packages/_rocm_sdk_devel
export HIP_PATH=$ROCM_PATH
INC="$ROCM_DEV_PATH/include"

# ----------------------------------------------------------------------
# Runtime + device library paths
# ----------------------------------------------------------------------
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$_ENV_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ROCM_DEV_PATH/lib:$_ENV_PREFIX/lib:$ROCM_PATH/lib:$_ENV_PREFIX/lib:$LIBRARY_PATH

export ROCM_DEVICE_LIB_PATH=$_ENV_PREFIX/lib/python$_PYVER/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
export HIP_DEVICE_LIB_PATH=$ROCM_DEVICE_LIB_PATH

# ----------------------------------------------------------------------
# Header include paths (JIT + thrust)
# ----------------------------------------------------------------------
export CPLUS_INCLUDE_PATH="$INC:${CPLUS_INCLUDE_PATH:-}"
export CPATH="$INC:${CPATH:-}"
export CXXFLAGS="-isystem $INC ${CXXFLAGS:-}"
export CPPFLAGS="-isystem $INC ${CPPFLAGS:-}"

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
echo "[INFO] ROCK runtime environment set:"
echo "       ROCM_PATH=$ROCM_PATH"
echo "       ROCM_DEV_PATH=$ROCM_DEV_PATH"
echo "       HIP_PATH=$HIP_PATH"
echo "       LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "       LIBRARY_PATH=$LIBRARY_PATH"
echo "       ROCM_DEVICE_LIB_PATH=$ROCM_DEVICE_LIB_PATH"
echo "       CPATH=$CPATH"
echo "       CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH"

# Quick thrust check
if [ -f "$INC/thrust/complex.h" ]; then
  echo "[CHECK] thrust: OK"
else
  echo "[CHECK] thrust: MISSING"
fi