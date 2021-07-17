#!/bin/bash
source config

export LOCAL_USER_ID=$(id -u)
#export YARP_DOCKER=${YARP_DOCKER}

# /Args/ $1: message
ask_user_to_proceed()
{
  echo "${1} [y/n]: "
  read CHOICE
  if [ ! ${CHOICE} = 'y' ]; then
    exit
  fi
}


# /Args/ $1: NVIDIA_CARD param
check_requirements()
{
  if [ ! `command -v docker` ]; then
    install_docker
  fi

  if [ ${1} = "yes" ]; then
    if [ ! `command -v nvidia-container-cli` ]; then
      install_nvidia_container
    fi
  fi

  if [ ! `command -v docker-compose` ]; then
    install_docker_compose
  fi

  echo "All Docker requirements are satisfied!"
}


install_docker()
{
  echo "Installing Docker on your Ubuntu system..."
  ask_user_to_proceed
  sudo apt-get update
  sudo apt install apt-transport-https ca-certificates curl software-properties-common
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu "$(lsb_release -cs)" stable"
  sudo apt update
  sudo apt install docker-ce
  echo "Adding "${USER}" to the docker group ..."
  ask_user_to_proceed
  sudo usermod -aG docker ${USER}
  echo "Please notice that you need to RESTART the machine in order these changes take effect!"
  echo "Press 'y' to reboot the PC or any other key to continue: "
  read CHOICE
  if [ ${CHOICE} = 'y' ]; then
    sudo reboot
  fi
}

install_nvidia_container()
{
  echo "Nvidia Docker is going to be installed in your Ubuntu system..."
  ask_user_to_proceed
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  echo "Restarting Docker ..."
  sudo systemctl restart docker;
}

install_docker_compose()
{
  echo "Installing docker-compose ..."

  if [ $NVIDIA_CARD = "yes" ]; then
    pip3 install --user git+https://github.com/beehiveai/compose.git
  else
    sudo curl -L "https://github.com/docker/compose/releases/download/1.26.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
  fi
  docker-compose --version
}


source config

if ! [ -x "$(command -v nvidia-settings)" ]; then
  NVIDIA_CARD="no"
else
  NVIDIA_CARD="yes"
fi

if [ $NVIDIA_CARD = "yes" ]; then
  COMPOSE_FILE="docker-compose-nvidia.yml"
else
  COMPOSE_FILE="docker-compose.yml"
fi

check_requirements $NVIDIA_CARD
