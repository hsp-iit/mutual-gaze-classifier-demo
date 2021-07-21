

FROM icubteamcode/human-sensing:master-stable_master_sources

MAINTAINER Elisa Maiettini

#### Install general dependencies ####
RUN apt-get update && apt-get install -y \
        software-properties-common

#RUN apt-get install -y  libgl1-mesa-dev libgles2-mesa-dev zlib1g-dev
RUN apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev \
    build-essential

#### Install python ####
#RUN apt-get install -y libmtdev-dev
RUN apt-get install -y python-dev
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
        python3.6 \
        python3-pip

RUN apt-get update && apt-get install -y \
        python3-distutils \
        python3-setuptools

RUN python3.6 -m pip install pip
RUN apt-get update

#### Install python packages ####
RUN python3.6 -m pip install --upgrade setuptools
RUN python3.6 -m pip install numpy==1.19.5 
RUN python3.6 -m pip install opencv-contrib-python==4.5.2.54
RUN python3.6 -m pip install pandas==1.1.5
RUN python3.6 -m pip install wheel==0.36.2
RUN python3.6 -m pip install scikit-learn==0.24.1

#### Install python bindings ####
RUN git config --global user.name icub
RUN git config --global user.email icub@iit.it

#### Install swig ####
RUN apt-get install swig

#### Compile yarp ####
RUN cd /projects/robotology-superbuild/build && \
    cmake .. \
    -DROBOTOLOGY_USES_PYTHON=ON \
    &&\
    make

RUN apt-get install -y nano gedit
RUN export PYTHONPATH=/projects/robotology-superbuild/build/src/YARP/lib/python3
#### Clone repository installation ####
ARG username
ARG password

RUN cd /projects/ && \
    git clone https://${username}:${password}@github.com/MariaLombardi/mutual-gaze-classifier-demo.git
RUN yarp namespace /root
RUN cd /projects/mutual-gaze-classifier-demo && \
    mkdir build && cd build && \
    cmake .. \
    &&\
    make \
    &&\
    make install

ENV CUDA_VISIBLE_DEVICES=1
