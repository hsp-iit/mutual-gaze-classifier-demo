#start from image passed by argument during build process. Usually it is an ubuntu image plus mesa library.
ARG START_IMG="none"
FROM $START_IMG

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies given in documentation in superbuild
# https://github.com/robotology/robotology-superbuild#linux
RUN mkdir -p /etc/bash_completion.d/ &&\
    apt-get update &&\
    apt-get install -y \
        # MISC
        bash-completion \
        qt5-default \
        git \
        cmake \
        cmake-curses-gui \ 
        build-essential \
        vim \
        #core
        coinor-libipopt-dev \ 
        libboost-system-dev \
        libboost-filesystem-dev \
        libboost-thread-dev \
        libopencv-dev \
        libode-dev \
        liblua5.1-dev \
        lua5.1 \ 
        swig \ 
        libsdl1.2-dev \
        libxml2-dev\
        # YARP
        libace-dev \
        libeigen3-dev \
        libgsl0-dev \
        libedit-dev \
        libqcustomplot-dev \
        qtmultimedia5-dev \
        qtdeclarative5-dev \
        libqt5opengl5-dev \
        qttools5-dev \
        libtinyxml-dev \
        #yarp-gui
        qtbase5-dev \
        qtdeclarative5-dev \
        qtmultimedia5-dev \
        qml-module-qtquick2 \
        qml-module-qtquick-window2 \
        qml-module-qtmultimedia \
        qml-module-qtquick-dialogs \
        qml-module-qtquick-controls \
        qml-module-qt-labs-folderlistmodel \
        qml-module-qt-labs-settings 

#In order to install cmake 12.0 required by superbuild
RUN apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget &&\
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - &&\
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' &&\
    apt-get update &&\
    apt-get install -y kitware-archive-keyring &&\
    apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA &&\
    apt-get install -y cmake \
    &&\
    rm -rf /var/lib/apt/lists/* 



#Some definitions
ARG PROJECTS_DIR=/projects
ARG CMAKE_GENERATOR="Unix Makefiles"
ARG BUILD_TYPE=Release
ARG CMAKE_EXTRA_OPTIONS=-j2
ARG INSTALL_DIR="/usr/local"
ARG release="master -b master_branch"
ARG sbtag="Stable"
#RUN if [ "$release" = "Stable" ]; then echo STABLE version is building; elif [ "$release" = "Ustable" ]; then echo UNSTABLE version is building && ${TAG}=Unstable; else echo SPECIFIC version $release is building && branch=$release; fi


RUN echo "DEBUG ==>  Release:" ${release} &&\
    echo "DEBUG ==>  TAG: " ${tag}


# Setup entrypoint
ARG ROBOTOLOGY_INITIALIZATION_FILE=/usr/local/bin/setup_robotology_tdd.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY setup.sh ${ROBOTOLOGY_INITIALIZATION_FILE}
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

RUN mkdir ${PROJECTS_DIR} && cd ${PROJECTS_DIR} &&\
    git clone https://github.com/robotology/robotology-superbuild.git &&\
    cd robotology-superbuild &&\
    git checkout ${release} &&\
    mkdir build && cd build &&\
    cmake .. \
        -G "$CMAKE_GENERATOR" \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DNON_INTERACTIVE_BUILD:BOOL=ON \
        -DROBOTOLOGY_ENABLE_CORE:BOOL=ON \
        -DYCM_USE_DEPRECATED:BOOL=OFF \
        -DROBOTOLOGY_USES_GAZEBO=OFF \
        -DROBOTOLOGY_USES_PYTHON=ON \
        -DROBOTOLOGY_PROJECT_TAGS=${sbtag} \
        -DYCM_EP_INSTALL_DIR=${INSTALL_DIR} \
        &&\
    # Build all the projects
    cmake --build . --target update-all -- -j4 &&\
    cmake --build . -- ${CMAKE_EXTRA_OPTIONS}


# This step ensures python looks for yarp bindings in the correct place
RUN echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3/dist-packages' >> $ROBOTOLOGY_INITIALIZATION_FILE


#copy scripts in /usr/local/bin
#RUN  cp ${PROJECTS_DIR}/robotology-superbuild/robotology/YARP/scripts/yarp_completion ${INSTALL_DIR}/bin  &&\
#RUN  mv ${PROJECTS_DIR}/robotology-superbuild/robotology/YARP/data/bash-completion/yarp ${INSTALL_DIR}/bin/yarp_completion  &&\
#       chmod +x ${INSTALL_DIR}/bin/yarp_completion

#RUN  cp ${PROJECTS_DIR}/robotology-superbuild/robotology/YARP/data/bash-completion/yarp ${INSTALL_DIR}/bin &&\
#        chmod +x ${INSTALL_DIR}/bin/yarp

RUN echo "source ${INSTALL_DIR}/share/robotology-superbuild/setup.sh" >> $ROBOTOLOGY_INITIALIZATION_FILE


# Waiting https://github.com/robotology/robotology-superbuild/issues/95
RUN echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROBOTOLOGY_SUPERBUILD_INSTALL_PREFIX}/lib/rtf'  >> $ROBOTOLOGY_INITIALIZATION_FILE

# The bashrc is read only when opening an interactive shell. Let other projects find packages contained in the superbuild.
ENV CMAKE_PREFIX_PATH=${INSTALL_DIR}

#add checkRobotInterface
RUN cd ${PROJECTS_DIR} &&\
    git clone https://github.com/icub-tech-iit/appsAway.git && \
    cd  appsAway/modules/checkRobotInterface && \
    mkdir build && cd build && \
    cmake .. \
        -G "$CMAKE_GENERATOR" \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        &&\
    cmake --build . -- ${CMAKE_EXTRA_OPTIONS} && \
    cp ./checkRobotInterface /usr/local/bin

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

RUN echo 'export img_metadata=${metadata}'  >> $ROBOTOLOGY_INITIALIZATION_FILE
RUN echo 'echo 'This images has release=$release and had been building with superbuild_tag=$sbtag. Metadata=$metadata ' '  >> $ROBOTOLOGY_INITIALIZATION_FILE

CMD ["bash"]


RUN apt-get update && apt-get install -y \
        software-properties-common
RUN apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    zlib1g-dev

#### Starts custom ####

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

RUN apt-get update

#### Install python packages ####
RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install --upgrade setuptools
RUN python3.6 -m pip install numpy
RUN python3.6 -m pip install opencv-contrib-python
RUN python3.6 -m pip install pandas
RUN python3.6 -m pip install wheel
RUN python3.6 -m pip install scikit-learn
RUN python3.6 -m pip install keras
RUN python3.6 -m pip install tensorflow
RUN python3.6 -m pip install pynput

#### Dependencies ####
RUN apt-get install -y libgoogle-glog-dev libboost-all-dev libhdf5-serial-dev libatlas-base-dev

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
    make -j12
RUN export PYTHONPATH=/projects/robotology-superbuild/build/src/YARP/lib/python3


#### Install openpose ####
RUN cd /projects &&\
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose &&\
    cd openpose/ &&\
    git submodule update --init --recursive --remote && \
    git checkout tags/v1.6.0

# compile and install openpose
RUN  apt-get -y install protobuf-compiler
RUN cd /projects/openpose &&\
    mkdir build
RUN cd /projects/openpose/build &&\
    cmake .. -DDOWNLOAD_HAND_MODEL=OFF
#    cmake .. -DDOWNLOAD_HAND_MODEL=OFF -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN=75 -DCUDA_ARCH_PTX=75
RUN cd /projects/openpose/build &&\ 
    make -j$(nproc) &&\
    make install

# compile and install yarpOpenPose
RUN cd /projects && git clone https://github.com/robotology/human-sensing
RUN cd /projects/human-sensing/yarpOpenPose &&\
    mkdir build && \
    cd /projects/human-sensing/yarpOpenPose/build &&\
    cmake .. &&\
    make && make install

#### Clone repository installation ####
RUN cd /projects/ && \
    git clone https://github.com/MariaLombardi/mutual-gaze-classifier-demo.git
RUN cd /projects/mutual-gaze-classifier-demo && \
    mkdir build && cd build && \
    cmake .. \
    &&\
    make \
    &&\
    make install \

RUN yarp namespace /cer04
RUN yarp conf 192.168.100.10 10000
