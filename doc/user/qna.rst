========================
Q&A
========================

How to compile my application?
===============================
We highly recommend using Clion.
If you are using Clion, add :code:`-DRAI_APP=<your appfile location in applications>` in the cmake flag.
For example, :code:`-DRAI_APP=examples/poleBalwithDDPG`.

If you are not using Clion, then::

    cd $RAI_ROOT
    mkdir build && cd build
    cmake .. -DRAI_APP=<your appfile location in applications>
    make -j

RAI is an ordinary cmake project. All you have to worry is to set the RAI_APP flag in cmake.

How to add a new application?
==================================
The best way is to copy one of the example folder to another folder in :code:`application` and rewrite :code:`CMakeLists.txt` and the main cpp file.
The outermost :code:`CMakeLists.txt` file will include your application folder once you set RAI_APP flag in cmake.

How to add a new task?
========================
Your task inherits from Task class in :code:`RAI/include/rai/tasks/Task.hpp`.
Copy :code:`RAI/include/rai/tasks/newTaskTemplate/newTask.hpp` to a new task folder and modify.

How to add a new algorithm?
==============================
Algorithm classes do not inherit any class and you can write in freely.
The best practice is to incorporate already existing core modules without reinventing them.

How to add a new neural net?
===============================
Neural net shapes are called graph structure in RAI.
:code:`RAI/include/rai/function/tensorflow/pythonProtobufGenerator/graph_structure` contains a few examples.
Make sure that you collect 'learnable parameters' in :code:`self.l_param_list` and 'all parameters' in :code:`self.a_param_list`.
Note that 'learnable parameters' is a subset of 'all parameters'.
If you are planning to make a commonly used graph_structure, send a pull request and include the figures as well.

How to add a new specialized function?
========================================
If your algorithm use something other than Qfunction, Valuefunction and Policy, you can implement a new specialized function.
Copy existing one (e.g. ValueFunction) and modify it. Make use that you write :code:`input_names` and :code:`output_names`.
If you have extra parameters to learn, simply add them to :code:`gs.l_param_list` and :code:`gs.a_param_list` (GraphStructure objects) before you call :code:`__init__` of the parent class.
:code:`l_param_list` means all learnable parameters and :code:`a_param_list` is all parameters. RAI makes a clear distinction between them.

Does RAI support multi-threading?
==================================
RAI supports multi-threaded data collection using parallel experience acquisitor, which is based on openmp [1]_.
openmp will automatically choose the number of cores to use.
If you want to specify it explicitly, call :code:'omp_set_num_threads(nThread)` in the beginning of your app file.
Multi-gpu support is not implemented yet.

Does RAI have a cluster-support?
==================================
Not yet.

Does RAI support OpenAI Gym?
==================================
Not yet.

Who is developing RAI?
========================
Jemin Hwangbo, Robotic System Laboratory, ETH Zurich.

How can I contribute?
========================
Send a pull request on the bitbucket repo. Please be specific when you explain the new features or bug fixes.

========================
Known Issues
========================

Compatibility with Gazebo
============================
Tensorflow install from source also installs a specific version of libprotobuf. This can cause Gazebo to malfunction.
The only work around we found is to install Gazebo from source.


.. [1] http://www.openmp.org/
