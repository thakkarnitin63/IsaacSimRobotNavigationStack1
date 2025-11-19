#!/bin/bash
# setup_ros_env.sh

# 1. Check if ISAACSIM_PATH is set
if [ -z "$ISAACSIM_PATH" ]; then
    echo "Error: \$ISAACSIM_PATH environment variable is not set."
    return 1
fi

# 2. Unset PYTHONPATH
# Prevents conflict with system's Python 3.10 ROS 2
unset PYTHONPATH
echo "-> PYTHONPATH unset."

# 3. Prepend the internal ROS 2 Bridge C++ libraries
# Fixes the 'ROS2 Bridge startup failed' error
BRIDGE_LIB_PATH="$ISAACSIM_PATH/exts/omni.isaac.ros2_bridge/humble/lib"
export LD_LIBRARY_PATH="$BRIDGE_LIB_PATH:$LD_LIBRARY_PATH"
echo "-> LD_LIBRARY_PATH updated."

echo "✅ Isaac Sim ROS 2 environment is ready."