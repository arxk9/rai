========================
Installation
========================

There are two existing installation scripts. First you have to run :code:`install_basics.sh` and install all basic dependencies.
If you are going to use GPU for tensorflow, install cuda8 and cudnn6.
Then, you can run :code:`install_tensorflow.sh` to install tensorflow from source.

Tensorflow will ask :code:`Please specify the location of python. [Default is /usr/bin/python]:`. The path you have to chose is printed two lines above the question.
Just copy paste the directory. The rest of the questions can be set as default except cuda related questions, depending on if you want to use cuda or not.

If you have a previous tensorflow installation from source, this might create a conflict depending on the version of the existing tensorflow.
The most common issue is protobuf version mismatch (this always occurs even for fresh install) which can be resolved by reinstalling protobuf as follows::

    cd /home/<yourID>/.cache/bazel/_bazel_root/<commit number>/external/protobuf/
    sudo ./autogen.sh && sudo ./configure && sudo make -j3 && sudo make install

Another problem we often face is::

    ImportError: libcudnn.Version: cannot open shared object file: No such file or directory

This is due to wrong cudnn version. Install cudnn6 and run :code:`sudo ldconfig -v` in the terminal followed by :code:`source ~/.bashrc`. This should solve the issue.

There are two ways to use RAI: 1. By adding your application in the application folder just like the example applications (recommended). 2. By building RAI library and linking it to your project.
To do the second option install as::

    mkdir build && cd build && cmake .. && sudo make install -j

This locally installs RAI in the build folder but installs a few files globally so that you can find RAI in your project.
To link RAI in cmake::

    find_package(RAI REQUIRED)
    include_directories(${RAI_INCLUDE_DIR})
    target_link_libraries(<YOUR APP> ${RAI_LINK} )

One problem that is not resolved now is that Tensorflow needs more recent version than usual ubuntu default.
One way is just to use Eigen version provided by Tensorflow by :code:`include_directories(${TENSORFLOW_EIGEN_DIR})`.
Another way is just install the same eigen version system-wise.
