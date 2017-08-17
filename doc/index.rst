.. RAI documentation master file, created by
   sphinx-quickstart on Mon May 29 16:27:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is RAI?
===============================

RAI is currently in beta testing stage. Feel free to contact jhwangbo@ethz.ch for suggesting possible improvements.

RAI is a C++ framework to develop and benchmark learning algorithms.
It is designed for reinforcement tasks with robotic applications in mind.
It only works in Linux systems and we do not have a plan to support Windows yet.
RAI is just a collection of classes that are useful for learning.
They are categorized into the core module and non-core modules.
The core module contains classes that are essential for reinforcement learning (e.g. algorithms, noise, tasks, memory, etc) and non-core modules contain useful tools for codding (e.g. a bunch of utils and graphics).
Most files are header-only for simplicity.
The easiest way to get started is to take a look at the example file presented in this tutorial.

What you need to know
===============================
We only use the word "state" instead of "observation" since many people out side of RL community is not familiar with the term "observation".
We assume that the goal is to minimize the discounted return and no other types of goals are supported yet (the discount can be set to zero though).

Basic User Guide
===============================

We will walk your through a set of tutorials that you must know in oder to use basic features of RAI.

.. toctree::
   :maxdepth: 3

   user/installation
   user/simple_example
   user/core
   user/rai_modules
   user/qna

Cite RAI
===============================
Please cite the following paper

@article{hwangbo2017control,
  title={Control of a Quadrotor with Reinforcement Learning},
  author={Hwangbo, Jemin and Sa, Inkyu and Siegwart, Roland and Hutter, Marco},
  journal={IEEE Robotics and Automation Letters},
  year={2017},
  publisher={IEEE}
  }
