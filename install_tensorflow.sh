#!/bin/bash

### General configuration of bash and flags for apt-get and add-apt-repository
set -e
set -o xtrace

APT_GET_FLAGS=-qq
ADD_APT_REPOSITORY_FLAGS=-y

########### installed CUDA and CudNN here ##############
### CUDA by downloading deb file and follow instructions on download page
### Do not change NVidia version since cuda installer will automatically update it to what it needs
### CudNN by downloading CudNN file and copy paste it to cp cuda/include/* usr/local/cuda/include/ and cp cuda/lib64/* usr/local/cuda/lib64/

#sudo add-apt-repository ppa:fkrull/deadsnakes
#sudo apt-get update
#sudo apt-get install python3.5

# Enable 
source ~/.bashrc
VIRTUALENVWRAPPER_PYTHON='/usr/bin/python3'
source virtualenvwrapper.sh

# Generate virtualenv for tensorflow (called tensorflow)
VIRTUALENV_NAME=tensorflow
rm -rf "$WORKON_HOME/$VIRTUALENV_NAME"
mkvirtualenv -p python3.5 $VIRTUALENV_NAME || true

# Activate virtualenv
workon $VIRTUALENV_NAME

# Installing python-dev
sudo apt-get install $APT_GET_FLAGS python-dev
sudo -H pip3 install numpy

##### now run configure
cd "$RAI_ROOT"
cd ./deepLearning
sudo rm -rf tensorflow
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.3
echo "select the following path as your python path: " $WORKON_HOME/tensorflow/bin/python
sudo ./configure

sed -i '\@https://github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz@d' tensorflow/workspace.bzl

echo -n "do you have gpu (y/n)? "
read answer
if echo "$answer" | grep -iq "^y" ;then
    sudo bazel build -c opt --config=cuda --copt="-mtune=native" --copt="-O3" tensorflow:libtensorflow_cc.so tensorflow:libtensorflow.so --genrule_strategy=standalone --spawn_strategy=standalone
    pip3 install --upgrade tensorflow-gpu
else
    sudo bazel build -c opt --copt="-mtune=native" --copt="-O3" tensorflow:libtensorflow_cc.so tensorflow:libtensorflow.so --genrule_strategy=standalone --spawn_strategy=standalone
    pip3 install --upgrade tensorflow
fi

# Update protobuf
cd $HOME/.cache/bazel/_bazel_root
for d in */ ; do
    if [ "$d" != "install/" ]; then
	echo "Entering $d" 
	cd $d/external/protobuf
	sudo ./autogen.sh && sudo ./configure && sudo make -j3 && sudo make install
    fi
done

exit



