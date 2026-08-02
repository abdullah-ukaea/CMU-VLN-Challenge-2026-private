FROM zhangjicmu/ubuntu24_ros:ai_module

# Copy the competition AI workspace into the existing ROS 2 workspace.
COPY src /home/docker/ai_module/src

# Build the official dummy and the Q-MapNav package against the base workspace.
WORKDIR /home/docker/ai_module
RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select dummy_vlm qmapnav"

# Ensure the updated install is sourced in every new shell
RUN echo "source /home/docker/ai_module/install/setup.bash" >> /home/docker/.bashrc
