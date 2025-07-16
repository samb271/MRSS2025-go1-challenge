#!/bin/bash

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source workspace
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

# Execute command
exec "$@"
