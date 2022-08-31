
# Equivariant Action Maps

This is the repository for the Spatially Equivariant Action Maps MSc thesis project. Paper link will be added.

Spatially Equivariant Action Maps (EAM) is a novel action representation for Visual Robot Learning that aims to leverage the equivariant
properties of SE(2) grasping space. Specifically, it maps final end-effector positions of a robot arm into a 2D
numpy array and assigns each position a weight. The element with the max position indicates where the end-effector will move towards.

<p>
    <img src="git_images/EAM.drawio-2.png" width=50% height=50% />
</p>

The generated 2D matrix is the same size and the visual input and the pixel values are aligned with the workspace.
This allows the representation to transform like the input image, meaning the input and the output are equivariant.

<p>
    <img src="git_images/EAM Eq.drawio.png" width=50% height=50% />
</p>

The final model proposed in the paper uses an FCN network to generate the final EAMs. The output maps need to be
passed through an interpretation layer with the workspace positions. This EAM architecture is found below:

<p>
    <img src="git_images/FullEAMDiag.drawio.png" width=75% height=75% />
</p>

Within the code about "rotated" EAMs, you can find an updated implementation of EAMs where a new rotational network is used
to determine the rotation of the end-effector in order to grasp a given object. It uses the positional argument from an EAM to 
crop a part of the input and generate a rotation vector. However, this is not the a part of the proposed work in the thesis.

<p>
    <img src="git_images/Rot Fig.drawio.png" width=75% height=75% />
</p>


## Requirements
This project was built on [PyRep](https://github.com/stepjam/PyRep) and [Coppelia Sim](https://www.coppeliarobotics.com). 
Refer to the installation requirements of each.

The code is written in Python and requires PIL, PyTorch, and [e2cnn](https://github.com/QUVA-Lab/e2cnn) libraries to work.

## Structure and How to Run
The files are structured in the following way:

* **Model Outlines:** This folder holds files that have the PyTorch ML networks used in the project. *Action Models* refer to the models used to train EAMs while the rest refer to models used to train end-effector velocity prediction models.
* **Simulations:** This folder holds the simulation environments that data_gathering and testing runs. Required by PyRep and CoppeliaSim.
* **Test Sequeunces:** This folder holds the code for various testing modules built for the project. To test EAMs, use *"test_action_rotation"*.
* **Train Sequences:** This folder holds the code for various training modules built for the project. To train EAMs, use *"train_action"* and to train the crop rotation model use*"train_action_rotation"*.
* Gather data is spread across the main directory. For EAMs, use *"gather_action_area"* or *"gather_action_area_rotation"*.

Within each Train, Test, and Gather Data modules, certain parameters are written in CAPITAL LETTERS after importing relevant libraries.
These provide options to adjust the training parameters, testing environment and parameters, and data gathering parameters.

##Extra Information
Below are the two suggested EAM inputs for training. This can be switched using the gradient option in the gather_data file.

<p>
    <img src="git_images/EAM Types.drawio-2.png" width=50% height=50% />
</p>

Finally, you can find some generated results of EAM models trained without rotating rectangles tested without rotation, with 90 degree rotation, and with random rotations.
The images presented are in that order.

**EAMs generated for non-rotated rectagles**
<p>
    <img src="git_images/Vert Examples.drawio.png" width=50% height=50% />
</p>
**EAMs generated for 90-degree rotated rectagles**
<p>
    <img src="git_images/90 Examples.drawio.png" width=50% height=50% />
</p>
**EAMs generated for randomly rotated rectagles**
<p>
    <img src="git_images/Turn Examples.drawio.png" width=50% height=50% />
</p>
