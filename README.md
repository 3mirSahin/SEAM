
# Equivariant Action Maps

This is the repository for the Equivariant Action Maps MSc thesis project. Paper link will be added.

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

