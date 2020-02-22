FROM gcr.io/kaggle-images/python:v56

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install dependences
RUN apt-get update --fix-missing && \
  apt-get install -y \
    wget \
    bzip2 \ 
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    mercurial \
    subversion \
    sudo \
    git \
    zsh \
    openssh-server \
    wget \
    gcc \
    g++ \
    libatlas-base-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    curl \
    make \
    unzip \
    # MeCab
    swig mecab libmecab-dev mecab-ipadic-utf8 \
    cmake --fix-missing

RUN pip install -U pip && \
    pip install fastprogress japanize-matplotlib neologdn && \
    pip install mecab-python3 kaggler
