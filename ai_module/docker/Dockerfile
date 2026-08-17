FROM zhangjicmu/ubuntu24_ros:ai_module

# Install the bounded detector runtime. PyTorch is pinned to a
# Python 3.12 CUDA wheel; the host driver may be newer because NVIDIA drivers
# retain compatibility with older CUDA runtimes.
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
      --break-system-packages \
      --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.7.1 \
      torchvision==0.22.1

COPY requirements-detectors.txt /tmp/qmapnav-requirements-detectors.txt
RUN python3 -m pip install \
      --break-system-packages \
      --no-cache-dir \
      -r /tmp/qmapnav-requirements-detectors.txt

# The selected detector must not download weights during an episode.
# Keep both its visual checkpoint and prompt encoder in a fixed image path.
RUN mkdir -p /home/docker/models
WORKDIR /home/docker/models
RUN python3 -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('yoloe-11s-seg.pt'); attempt_download_asset('mobileclip_blt.ts')"

USER docker

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
