FROM nvcr.io/nvidia/isaac-sim:4.5.0

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-nav-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install \
    opencv-python \
    numpy \
    scipy \
    pyyaml \
    gymnasium

# Install Isaac ROS packages
RUN apt-get update && apt-get install -y \
    ros-humble-isaac-ros-common \
    ros-humble-isaac-ros-visual-slam \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS environment
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set up Isaac Sim environment
ENV ISAAC_SIM_ROOT=/isaac-sim
ENV PYTHONPATH="${ISAAC_SIM_ROOT}/kit/python/lib:${PYTHONPATH}"
ENV PYTHONPATH="${ISAAC_SIM_ROOT}/extscache/omni.pip.torch-1_13_1-cp310-cp310-linux_x86_64/torch-1.13.1+cu117-cp310-cp310-linux_x86_64.whl/torch:${PYTHONPATH}"

# Create workspace
WORKDIR /workspace
COPY . /workspace/

# Build ROS packages
RUN cd /workspace && \
    /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
