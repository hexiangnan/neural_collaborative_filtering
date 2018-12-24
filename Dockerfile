FROM ubuntu:16.04

ENV KERAS_BACKEND theano

RUN mkdir -p /tmp/setup && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python-dev \
        python-pip \
        python-setuptools \
        software-properties-common \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --user pip==9.0.3 && \
    pip install Theano==0.8.0 && \
    pip install keras==1.0.7 && \
    pip install h5py && \
    pip install numpy==1.11.0

WORKDIR /home
CMD ["/bin/bash"]
