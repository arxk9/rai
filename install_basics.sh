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

### Check Ubuntu version 
version=$(lsb_release -r | awk '{ print $2 }')
yrelease=$( echo "$version" | cut -d. -f1 )
mrelease=$( echo "$version" | cut -d. -f2 )

### General paths
sed -i "/\b\RAI_ROOT\\b/d" $HOME/.bashrc
printf 'export RAI_ROOT='$PWD'\n' >> $HOME/.bashrc
RAI_ROOT="$PWD"

cd "$RAI_ROOT"
mkdir -p dependencies

### adding ppa's
# compilers
sudo add-apt-repository $ADD_APT_REPOSITORY_FLAGS ppa:ubuntu-toolchain-r/test

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

### Bazel
#sudo apt-get install $APT_GET_FLAGS software-properties-common
#sudo apt-get install $APT_GET_FLAGS golang
#
#if [ "$yrelease" -eq "16" ]; then
#	sudo apt-get install $APT_GET_FLAGS openjdk-8-jdk
#else
#	if [ "$yrelease" -eq "14" ]; then
#		sudo add-apt-repository ppa:webupd8team/java
#		sudo apt-get update  $APT_GET_FLAGS && sudo apt-get install $APT_GET_FLAGS oracle-java8-installer
#	fi
#fi
#
#echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
#curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
#sudo apt-get update $APT_GET_FLAGS 
#sudo apt-get install $APT_GET_FLAGS bazel
#sudo apt-get upgrade $APT_GET_FLAGS bazel

### Bazel 0.5.4
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh
chmod +x bazel-0.5.4-installer-linux-x86_64.sh
sudo ./bazel-0.5.4-installer-linux-x86_64.sh --prefix=/usr
rm bazel-0.5.4-installer-linux-x86_64.sh

printf 'export PATH="$PATH:$HOME/bin"\n' >> $HOME/.bashrc

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
sudo apt-get install $APT_GET_FLAGS python3-pip
pip3 install --upgrade pip

# Installing virtualenv
sudo apt-get install $APT_GET_FLAGS python-virtualenv
sudo apt-get install $APT_GET_FLAGS python3-setuptools

# Installing virtualenvwrapper
pip3 install $APT_GET_FLAGS virtualenvwrapper
sed -i '/WORKON_HOME/d' $HOME/.bashrc
printf 'export WORKON_HOME=~/.virtualenvs\n' >> $HOME/.bashrc
sed -i '/VIRTUALENVWRAPPER_PYTHON/d' $HOME/.bashrc
printf 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3\n' >> $HOME/.bashrc
sed -i '/source virtualenvwrapper.sh/d' $HOME/.bashrc
printf 'source virtualenvwrapper.sh\n' >> $HOME/.bashrc
source ~/.bashrc

### graphic
# 3D rendering
sudo apt-get install $APT_GET_FLAGS libglew-dev freeglut3-dev libsdl2-dev libglm-dev glee-dev libsdl2-image-dev libassimp-dev libsoil-dev libfreeimage3 libfreeimage-dev libsdl2-ttf-dev

# ffmpeg
if [ "$yrelease" -eq "16" ]; then
    sudo apt-get install $APT_GET_FLAGS ffmpeg
else
    if [ "$yrelease" -eq "14" ]; then
        sudo add-apt-repository ppa:mc3man/trusty-media
	sudo apt-get update $APT_GET_FLAGS 
	sudo apt-get dist-upgrade $APT_GET_FLAGS 
	sudo apt-get install $APT_GET_FLAGS ffmpeg
    fi
fi

# GNUPLOT
if [ "$yrelease" -eq "16" ]; then
    sudo apt-get install $APT_GET_FLAGS gnuplot5
else 
    if [ "$yrelease" -eq "14" ]; then
     cd $RAI_ROOT/dependencies
     # plotting dependencies
     sudo apt-get install $APT_GET_FLAGS libqt4-dev
     sudo apt-get install $APT_GET_FLAGS libcairo2-dev
     sudo apt-get install $APT_GET_FLAGS libpango1.0-dev
     wget https://downloads.sourceforge.net/project/gnuplot/gnuplot/5.0.5/gnuplot-5.0.5.tar.gz
     tar -xvzf gnuplot-5.0.5.tar.gz
     rm gnuplot-5.0.5.tar.gz
     cd gnuplot-5.0.5/
     ./configure --prefix=/usr --disable-wxwidgets && sudo make all -j && sudo make install
     cd $RAI_ROOT
    fi
fi

# Box2D
sudo apt-get install $APT_GET_FLAGS libbox2d-dev

# URDF parser
sudo apt-get install $APT_GET_FLAGS liburdfdom-dev

# INSTALL rbdl
cd $RAI_ROOT/dependencies
rm -rf rbdl
sudo apt-get install $APT_GET_FLAGS mercurial
hg clone https://bitbucket.org/rbdl/rbdl
cd rbdl/
mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=Release ../ && make -j && sudo make install

# RAI_Common
cd $(dirname "$RAI_ROOT")
#git clone git@bitbucket.org:jhwangbo/raicommon.git
git clone https://bitbucket.org/jhwangbo/raicommon.git
cd raicommon
cmake CMakeLists.txt && sudo make install -j

# RAI_Graphics
cd $(dirname "$RAI_ROOT")
#git clone git@bitbucket.org:jhwangbo/raigraphics_opengl.git/
git clone https://bitbucket.org/jhwangbo/raigraphics_opengl.git
cd raigraphics_opengl
sudo ./install.sh
cmake CMakeLists.txt && sudo make install -j

cd $RAI_ROOT
exit
