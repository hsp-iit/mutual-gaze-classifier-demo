ARG ICUB_DOCKER

FROM ${ICUB_DOCKER}
LABEL maintainer="Davide De Tommaso <davide.detommaso@iit.it>"

ARG LOCAL_USER_ID
ARG PSYCHOPY_VER

ENV DEBIAN_FRONTEND noninteractive

USER root

RUN usermod -u ${LOCAL_USER_ID} docky

# install cuda-toolkit
RUN apt-get install -y cuda-toolkit-10-0

# install cudnn
RUN apt-get install -y libcudnn7=7.6.1.34-1+cuda10.0
RUN apt-get install -y libcudnn7-dev=7.6.1.34-1+cuda10.0

# further libraries for cmake
RUN apt-get install -y nano
RUN apt-get install -y protobuf-compiler libprotobuf-dev
RUN apt-get install -y libgoogle-glog-dev libboost-all-dev libhdf5-serial-dev libatlas-base-dev

#SDK for the realsense
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
RUN apt-get install -y librealsense2-dkms librealsense2-utils
RUN apt-get install -y librealsense2-dev librealsense2-dbg

# add user docky to video group (use webcam)
RUN usermod -a -G video docky


USER docky

# clone openpose
RUN cd $HOME &&\
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose &&\
    cd openpose/ &&\
    git submodule update --init --recursive --remote
# compile and install openpose
RUN cd $HOME/openpose &&\
    mkdir build
RUN cd $HOME/openpose/build &&\
    cmake .. -DCMAKE_INSTALL_PREFIX=/home/docky/robotology-build -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN=75 -DCUDA_ARCH_PTX=75
RUN cd $HOME/openpose/build &&\ 
    make -j$(nproc) &&\
    make install

# compile and install yarpOpenPose
RUN cd $HOME/human-sensing/yarpOpenPose &&\
    mkdir build
RUN cd $HOME/human-sensing/yarpOpenPose/build &&\
    cmake .. -DCMAKE_INSTALL_PREFIX=/home/docky/robotology-build &&\
    make && make install

