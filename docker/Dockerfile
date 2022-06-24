FROM nvidia/cuda:11.2.0-runtime-ubuntu18.04

# Update and install mini-conda
RUN apt-get update
RUN apt-get install -y time wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 libboost-all-dev git mercurial subversion && \
        apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY ./docker/requirements.yaml /tmp/requirements.yaml
RUN conda update conda \
    && conda env create --name hd-ms-cluster -f /tmp/requirements.yaml

RUN echo "conda activate hd-ms-cluster" >> ~/.bashrc
ENV PATH /opt/conda/envs/hd-ms-cluster/bin:$PATH
ENV CONDA_DEFAULT_ENV $hd-ms-cluster

# Git the code and build
RUN git clone https://gitlab.nrp-nautilus.io/wh-xu/hd-ms-cluster.git /hd-ms-cluster

# set working dir
WORKDIR /hd-ms-cluster

# Build Cython 
RUN bash ./build.sh 
