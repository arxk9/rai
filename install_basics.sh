#!/bin/bash


# Jemin Hwangbo May 30, 2017
#
# Works on Ubuntu 14.04 and 16.04
# for Ubuntu 14.04, you need to add ffmpeg manually
# Register to bitbucket first, add ssh key then continue

### General configuration of bash and flags for apt-get and add-apt-repository
set -e
set -o xtrace

APT_GET_FLAGS=-qq
ADD_APT_REPOSITORY_FLAGS=-y

### General paths
sed -i '/RAI_ROOT/d' $HOME/.bashrc
printf 'export RAI_ROOT='$PWD'\n' >> $HOME/.bashrc
RAI_ROOT="$PWD"

cd "$RAI_ROOT"

### adding ppa's
# compilers
sudo add-apt-repository $ADD_APT_REPOSITORY_FLAGS ppa:ubuntu-toolchain-r/test
#bazel
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
#python
sudo add-apt-repository $ADD_APT_REPOSITORY_FLAGS ppa:fkrull/deadsnakes

## update
sudo apt-get update

### wget and curl
sudo apt-get install $APT_GET_FLAGS wget curl

### create certificate for curl
sudo mkdir -p /etc/pki/tls/certs
sudo cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt


# basics
sudo apt-get install $APT_GET_FLAGS gcc-6 g++-6 libeigen3-dev libtinyxml-dev autoconf automake libtool curl make g++ unzip

### Build essentials
sudo apt-get install $APT_GET_FLAGS build-essential

### logging
sudo apt-get install $APT_GET_FLAGS libgflags-dev libgoogle-glog-dev

### Boost
sudo apt-get install $APT_GET_FLAGS libboost-all-dev

## Bazel
sudo apt-get install $APT_GET_FLAGS software-properties-common
sudo apt-get install $APT_GET_FLAGS golang
sudo apt-get install $APT_GET_FLAGS bazel
sudo apt-get upgrade bazel

## Cmake 3
wget https://cmake.org/files/v3.6/cmake-3.6.1-Linux-x86_64.sh
chmod +x cmake-3.6.1-Linux-x86_64.sh
sudo ./cmake-3.6.1-Linux-x86_64.sh --skip-license --prefix=/usr
rm cmake-3.6.1-Linux-x86_64.sh

## Swig
sudo apt-get install $APT_GET_FLAGS swig

## Setting up python and virtualenv

# python 3.5
sudo apt-get install $APT_GET_FLAGS python3.5-dev

# Installing pip
sudo apt-get install $APT_GET_FLAGS python-pip

# Installing virtualenv
sudo apt-get install $APT_GET_FLAGS python-virtualenv

# Installing virtualenvwrapper
sudo pip install $APT_GET_FLAGS virtualenvwrapper

### graphic
# 3D rendering
sudo apt-get install $APT_GET_FLAGS libglew-dev freeglut3-dev libsdl2-dev libglm-dev glee-dev libsdl2-image-dev libassimp-dev libsoil-dev libfreeimage3 libfreeimage-dev ffmpeg libsdl2-ttf-dev

# plotting
sudo apt-get install $APT_GET_FLAGS gnuplot5

# Box2D
sudo apt-get install $APT_GET_FLAGS libbox2d-dev

# URDF parser
sudo apt-get install $APT_GET_FLAGS liburdfdom-dev

# INSTALL rbdl
cd $RAI_ROOT/..
sudo apt-get install $APT_GET_FLAGS mercurial
hg clone ssh://hg@bitbucket.org/rbdl/rbdl
cd rbdl/
mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=Release ../ && make -j && sudo make install

# docs
cd $RAI_ROOT/doc
mkvirtualenv sphinx
workon sphinx
pip install sphinx sphinx-autobuild
pip install sphinx_rtd_theme
make html
deactivate

cd $RAI_ROOT
exit
