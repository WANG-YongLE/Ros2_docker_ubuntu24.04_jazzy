FROM ros:jazzy

# 基本工具
RUN apt update && apt install -y \
    git \
    python3-colcon-common-extensions \
    python3-pip

RUN mkdir -p /workspace
# ROS workspace
RUN echo "alias r='source /root/rebuild_colcon.rc'" >> /root/.bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
WORKDIR /workspace