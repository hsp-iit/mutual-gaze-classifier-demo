FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# FROM ubuntu:24.04
ARG NJOBS=6

LABEL org.opencontainers.image.title="ergoCub cartesian control"
LABEL org.opencontainers.image.description="Software to control in cartesian space the ergocub robot"
LABEL org.opencontainers.image.source="https://github.com/hsp-iit/ergocub-cartesian-control/docker/Dockerfile"
LABEL org.opencontainers.image.authors="Francesco Brand <francesco.brand@iit.it>"

# Use /bin/bash instead of /bin/sh
SHELL ["/bin/bash", "-c"]

# Non-interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive

# Set the locale
RUN apt update && \
    apt install -y -qq locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install essentials
RUN apt update && \
    apt install --no-install-recommends -y -qq apt-utils build-essential ca-certificates cmake cmake-curses-gui curl git glmark2 gnupg2 htop iputils-ping jq lsb-release mesa-utils nano psmisc sudo unzip vim wget zip \
    python3-venv python3-dev python3-pip python3-numpy ffmpeg libavcodec-dev libavdevice-dev libavformat-dev libavutil-dev && \
    rm -rf /var/lib/apt/lists/*

# Install additional dependencies for the robotology superbuild
RUN apt update && \
    git clone https://github.com/robotology/robotology-superbuild && cd robotology-superbuild && git checkout v2025.02.0 && bash scripts/install_apt_dependencies.sh && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf ../robotology-superbuild

# Create user with passwordless sudo
RUN useradd -l -G sudo -md /home/user -s /bin/bash -p user user && \
    sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers

# Switch to user
USER user

# Build robotology-superbuild
WORKDIR /home/user
RUN git clone https://github.com/robotology/robotology-superbuild && \
    cd robotology-superbuild && \
    git fetch -a && \
    git checkout v2025.02.0 && \
    git config --local user.name "user" && \
    git config --local user.email "user@email.com" && \
    mkdir build && cd build && \
    cmake -DROBOTOLOGY_ENABLE_CORE=ON -DROBOTOLOGY_USES_PYTHON=ON -DROBOTOLOGY_USES_GAZEBO=OFF ../ && \
    make -j$NJOBS

# This step ensures python looks for yarp bindings in the correct place
ARG ROBOTOLOGY_INITIALIZATION_FILE=/home/user/setup_robotology_tdd.sh
RUN echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3/dist-packages' >> $ROBOTOLOGY_INITIALIZATION_FILE

RUN echo "source /home/user/robotology-superbuild/build/install/share/robotology-superbuild/setup.sh" >> $ROBOTOLOGY_INITIALIZATION_FILE

# Waiting https://github.com/robotology/robotology-superbuild/issues/95
RUN echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROBOTOLOGY_SUPERBUILD_INSTALL_PREFIX}/lib/rtf'  >> $ROBOTOLOGY_INITIALIZATION_FILE

# The bashrc is read only when opening an interactive shell. Let other projects find packages contained in the superbuild.
ENV CMAKE_PREFIX_PATH=${INSTALL_DIR}

#add checkRobotInterface
# RUN cd ${PROJECTS_DIR} &&\
#     git clone https://github.com/icub-tech-iit/appsAway.git && \
#     cd  appsAway/modules/checkRobotInterface && \
#     mkdir build && cd build && \
#     cmake .. \
#         -G "$CMAKE_GENERATOR" \
#         -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
#         &&\
#     cmake --build . -- ${CMAKE_EXTRA_OPTIONS} && \
#     cp ./checkRobotInterface /usr/local/bin

# apparently tf_slim is not installed by default, so we install it
RUN pip install tf_slim

#The EXPOSE instruction does not actually publish the port. 
#It functions as a type of documentation between the person who builds the image and the person who runs the container, about which ports are intended to be published. 
#To actually publish the port when running the container, use the -p flag on docker run to publish and map one or more ports, or the -P flag to publish all exposed ports and map them to high-order ports.
EXPOSE 10000/tcp 10000/udp

# Some QT-Apps don't show controls without this
ENV QT_X11_NO_MITSHM 1
ENV YARP_COLORED_OUTPUT 1


ARG metadata="none"

ENV img_metadata=${metadata}

CMD ["bash"]


RUN sudo apt-get update && sudo apt-get install -y \
    software-properties-common libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev \
    libgoogle-glog-dev libboost-all-dev libhdf5-serial-dev libatlas-base-dev

#### Install python packages ####
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools
RUN python3 -m pip install "numpy<2.0.0"
RUN python3 -m pip install opencv-contrib-python
RUN python3 -m pip install pandas
RUN python3 -m pip install wheel
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install keras
RUN python3 -m pip install tensorflow
RUN python3 -m pip install pynput

RUN export PYTHONPATH=/home/user/robotology-superbuild/build/src/YARP/lib/python3

# ### Install torch, necessary for mmpose
RUN python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install mmpose dependencies
RUN pip install -U openmim && \
    python3 -m mim install mmengine && \
    python3 -m mim install "mmcv==2.0.0rc4" && \
    python3 -m mim install "mmdet>=3.0.0"

# Install mmpose
RUN git clone https://github.com/open-mmlab/mmpose.git \
    && cd mmpose \
    && pip install -r requirements.txt \
    && pip install -v -e .

# Install mmdeploy and onnx runtime
RUN pip install mmdeploy \
    && pip install mmdeploy-runtime-gpu \
    && pip install onnxruntime-gpu \
    && wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz \
    && tar -zxvf onnxruntime-linux-x64-gpu-1.15.1.tgz 

ENV ONNXRUNTIME_DIR=/home/user/onnxruntime-linux-x64-gpu-1.15.1

ENV CUDA_BIN_PATH=/usr/local/cuda

ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-11.7/compat/:$LD_LIBRARY_PATH

# MMdeploy sdk prerequisites
RUN git clone https://github.com/openppl-public/ppl.cv.git \
    && cd ppl.cv \
    && git checkout tags/v0.7.0 -b v0.7.0 \
    && ./build.sh cuda

ENV PPLCV_DIR=/home/user/ppl.cv

# Install mmdeploy sdk
RUN git clone -b main https://github.com/open-mmlab/mmdeploy.git --recursive --depth=1 \
    && cd /home/user/mmdeploy \
    && mkdir -p build && cd build \
    && cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DMMDEPLOY_TARGET_DEVICES=cuda \
        -DMMDEPLOY_TARGET_BACKENDS=ort \
        -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DCUDNN_DIR=${CUDNN_DIR} \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j8 && sudo make install

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# # Install c++ unit test framework
# RUN git clone https://github.com/catchorg/Catch2.git \
#     && cd Catch2 \
#     && cmake -Bbuild -H. -DBUILD_TESTING=OFF \
#     && cmake --build build/ --target install


RUN python3 -m pip install "numpy<2.0.0"
ENV PYTHONPATH=$PYTHONPATH:/home/user/mmpose

# Install the application
RUN source /home/user/robotology-superbuild/build/install/share/robotology-superbuild/setup.sh && \
    git clone https://github.com/robotology/human-sensing.git \
    && cd human-sensing/yarpRTMPose \
    && mkdir build && cd build \
    && cmake .. \
    && make \
    && sudo make install \ 
    && cd ~/mmdeploy \
    && /home/user/human-sensing/yarpRTMPose/app/scripts/deploy.sh

#### Clone repository installation ####
RUN source /home/user/robotology-superbuild/build/install/share/robotology-superbuild/setup.sh && \
    git clone -b fix/Docker https://github.com/fbrand-new/mutual-gaze-classifier-demo.git && \
    cd mutual-gaze-classifier-demo && \
    mkdir build && cd build && \
    cmake .. && \
    sudo make install

# # RUN yarp namespace /cer04
RUN source /home/user/robotology-superbuild/build/install/share/robotology-superbuild/setup.sh && \
    yarp conf 192.168.100.10 10000
