# Our base OS Image
ARG BASE_IMAGE=nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
ARG PYTHON_VERSION=3.9
FROM ${BASE_IMAGE}

# FROM directive resets ARG
ARG BASE_IMAGE

# LABEL about the custom image
LABEL maintainer="gno320@gmail.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image for \
    running gpus env on Windows."

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# For creating non-root users
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEV_WORKSPACE=/home/$USERNAME/workspace

# Set timezone, options are America/Los_Angeles
ENV TZ=America/New_York

# adds anaconda to path
ENV PATH "/home/${USERNAME}/anaconda3/bin:$PATH"

# Create the user
RUN apt-get update -y \
    && apt-get install -y sudo tzdata \ 
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    # give the user ability to install software
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USER_UID}
ENV HOME=/home/${USERNAME}
WORKDIR ${DEV_WORKSPACE}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN sudo apt-get update -y && sudo apt-get upgrade -y \
    && sudo apt-get install -y \
        cmake \
        curl \
        ffmpeg \
        g++ \ 
        gcc \
        git \
        libgl1-mesa-dev \
        libgtk2.0-dev \
        libjemalloc-dev \
        python-opengl \
        qt5-default \
        unrar \
        unzip \
        vim \
        wget \
        zlib1g-dev \
        $(if [ "$AUTOSCALER" = "autoscaler" ]; then echo \
        tmux \
        screen \
        rsync \
        openssh-client \
        gnupg; fi) \
    && wget \
        --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
        -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -u -p $HOME/anaconda3 \
    && $HOME/anaconda3/bin/conda init \ 
    && echo 'export PATH=$HOME/anaconda3/bin:$PATH' >> /home/${USERNAME}/.bashrc \
    && rm /tmp/miniconda.sh \
    && $HOME/anaconda3/bin/conda install -y \
        libgcc python=$PYTHON_VERSION \
    && $HOME/anaconda3/bin/conda clean -y --all \
    && $HOME/anaconda3/bin/pip install --no-cache-dir \
        flatbuffers \
        cython==0.29.26 \
        # Necessary for Dataset to work properly.
        numpy\>=1.20 \
        psutil \
    # To avoid the following error on Jenkins:
    # AttributeError: 'numpy.ufunc' object has no attribute '__module__'
    && $HOME/anaconda3/bin/pip uninstall -y dask \ 
    # We install cmake temporarily to get psutil
    && sudo apt-get autoremove -y cmake zlib1g-dev \
        # We keep g++ on GPU images, because uninstalling removes CUDA Devel tooling
        $(if [ "$BASE_IMAGE" = "ubuntu:focal" ]; then echo \
        g++; fi) \
    # Either install kubectl or remove wget 
    && (if [ "$AUTOSCALER" = "autoscaler" ]; \
        then wget -O - -q https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - \
        && sudo touch /etc/apt/sources.list.d/kubernetes.list \
        && echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list \
        && sudo apt-get update \
        && sudo apt-get install kubectl; \
    else sudo apt-get autoremove -y wget; \
    fi;) \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

# Install dependencies for the environment
RUN $HOME/anaconda3/bin/pip --no-cache-dir install --upgrade pip \
    && ${HOME}/anaconda3/bin/pip --no-cache-dir install --upgrade \
    setuptools \
    'ray[rllib,tune]==1.13' \
    numpy \
    gputil \
    Pillow \
    h5py \
    matplotlib \
    cycler \
    tensorflow==2.6.2 \
    tensorflow_probability \
    keras==2.6 \
    jupyter \
    pandas \
    pybullet \
    stable_baselines3 \
    pyqt5 \
    qpsolvers[quadprog] \
    pyglet==1.5.27 \
    pybullet==3.2.0 \
    scikit-image \
    tensorboardX \
    torch \
    torch-tb-profiler \
    torchvision \
    torchaudio \
    wandb==0.12.5 \
    && if [ $(python -c 'import sys; print(sys.version_info.minor)') != "6" ]; then \
        $HOME/anaconda3/bin/pip uninstall dataclasses typing -y; fi 

# set up lightweight display, -E,-preserve-env ensures we get environment variables when using sudo
RUN sudo apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive sudo -E apt-get install -y x11vnc \
    xvfb \
    xfce4 \
    fonts-wqy-microhei fonts-wqy-zenhei \
    && sudo apt-get autoremove \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean \
    # create X11 socket directory
    && install -d -m 1777 /tmp/.X11-unix \
    && mkdir -p ~/.vnc \
    && x11vnc -storepasswd 1234 ~/.vnc/passwd \
    # Fix error with dubious workspace: Git detect dubious ownership in repository
    && sudo chown ${USERNAME} ${DEV_WORKSPACE}

# A script for loading the GUI app
COPY docker/entrypoint.sh /entrypoint.sh

# Starts the app. 
ENTRYPOINT ["/entrypoint.sh"]