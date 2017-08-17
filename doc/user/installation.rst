========================
Installation
========================

There are two existing installation scripts. First you have to run :code:`install_basics.sh` and install all basic dependencies.
If you are going to use GPU for tensorflow, install cuda8 and cudnn6.
Then, you can run :code:`install_tensorflow.sh` to install tensorflow from source.

Tensor flow will ask :code:`Please specify the location of python. [Default is /usr/bin/python]:`. The path you have to chose is printed two lines above the question.
Just copy paste the directory. The rest of the questions can be set as default except cuda related questions, depending on if you want to use cuda or not.

If you have a previous tensorflow installation from source, this might create a conflict depending on the version of the existing tensorflow.
The most common issue is protobuf version mismatch (this always occurs even for fresh install) which can be resolved by reinstalling protobuf as follows::

    cd /home/<yourID>/.cache/bazel/_bazel_root/<commit number>/external/protobuf/
    sudo ./autogen.sh && sudo ./configure && sudo make -j3 && sudo make install

