========================
Installation
========================

There are two existing installation scripts. First you have to run :code:`install_basics.sh` and install all basic dependencies.
If you are going to use GPU for tensorflow, install cuda8 and cudnn5.1 (Do not install cudnn 6 since Tensorflow does not support it yet).
Then, you can run :code:`install_tensorflow.sh` to install tensorflow from source.
If you have a previous tensorflow installation from source, this might create a conflict depending on the version of the existing tensorflow.
The most common issue is protobuf version mismatch (this always occurs even for fresh install) which can be resolved by reinstalling protobuf as follows::

    cd /home/<yourID>/.cache/bazel/_bazel_root/<commit number>/external/protobuf/
    sudo ./autogen.sh && sudo ./configure && sudo make -j3 && sudo make install

