# 1. Contributors
Main developer: Jemin Hwangbo (RSL, ETH Zurich, Switzerland, jhwangbo@ethz.ch)

Contributors: Junho Lee (ETH Zurich, Switzerland), Michael Ackermann (ETH Zurich, Switzerland)

# 2. Installation
Clone with "git clone --recursive {URL}"

(Optional) Install Nvidia driver, Cuda, Cudnn if you use GPU.

Run install.sh: *./install_basics.sh*

This installs all the dependencies except tensorflow

If you are going to use GPU, install cuda and cudnn from Nvidia website.

Then run *install_tensorflow.sh*

# 3. Recommended developing environment
+ Clion is recommended
+ Use google style (available in Clion by default) and 105 characters per line

# 4. Allowed abbreviations in the code
+ traj = trajectory
+ dim = dimension
+ term = terminal/termination
+ bat = batch
+ Fig = figure
+ prop = properties
+ LP/AP = learnable/all parameters

# 5. To cite RAI
Please cite the following paper

@article{hwangbo2017control,
title={Control of a Quadrotor with Reinforcement Learning}, author={Hwangbo, Jemin and Sa, Inkyu and Siegwart, Roland and Hutter, Marco}, journal={IEEE Robotics and Automation Letters}, year={2017}, publisher={IEEE} }
Next 





