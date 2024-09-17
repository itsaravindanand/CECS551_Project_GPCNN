# CECS551_Project_GPCNN

A repository of the project files of Parameter Reduction in CNNs: The Role of Grouped Pointwise Convolution
Project Setup:
To execute the given python jupyter notebook, few software’s are required, and the details are furnished in the below table along with the link to download and setup.

Project Report: [GPCNN](https://github.com/itsaravindanand/CECS551_Project_GPCNN/blob/main/CECS551_Project_Files/Parameter_Reduction_in_CNNs_Project_Report.pdf)

Project File: [GPCNN](https://github.com/itsaravindanand/CECS551_Project_GPCNN.git)

GPC-API: [gpc-api](https://github.com/itsaravindanand/gpc-api.git)

Experiment results:
| Architecture | Parameters Count | Parameter Reduction Percentage | Parameter Size | Computation (FLOPS) | Computation Reduction Percentage | Test Accuracy |
|---|---|---|---|---|---|---|
| MobileNet | 3.239M | NA | 12.36 MB | 567.751M | NA | 92.35% |
| gpcMobileNet (16 Channels) | 0.274M | 91.54% | 1.05 MB | 32.122M | 94.34% | 89.67% |
| gpcMobileNet (32 Channels) | 0.432M | 86.66% | 1.65 MB | 57.813M | 89.81% | 90.82% |
| gpcMobileNet (64 Channels) | 0.748M | 76.90% | 2.85 MB | 83.503M | 85.29% | 91.96% |
| gpcMobileNet (128 Channels) | 1.353M | 58.22% | 5.16 MB | 160.172M | 71.78% | 92.69% |

List of software/packages required and the respective versions:

## Software/Package | Version | Description
----------------------------------------------------------------------------
Anaconda Navigator	| 2.5.1	| Anaconda | Used to create Runnable Environment

Python	| 3.9.18	| Selected programming language to execute the project

Tensorflow	| 2.10.1	| Library with inbuilt packages to handle tensor values

Cuda	| 11.2.0	| Required to connect to NVIDIA GPU

Cudnn	| 8.1.0	|	Required to connect to NVIDIA GPU

Cuda Toolkit	| 12.3	| Required to connect to NVIDIA GPU

Cudnn Dependency Files	| NA	| Required to connect to NVIDIA GPU

git	| NA	| Version Control Sytem required to clone the dependency files

Jupyter Notebook	| NA	| Required to execute the dependency files and evaluate the results

----------------------------------------------------------------------------

Note:
-	A step wise guide for installing anaconda navigator and tensorflow can be found in this [link](https://www.tensorflow.org/install/pip#windows-native_1).
-	The links specified are subject to update.
-	The python notebook was executed with the software/packages with specification mentioned above. When the files are executed in another setup, a slight difference in the accuracy values is expected.
-	The execution time differs based on the hardware setup used. The current outputs in the python notebook were executed with a batch size of 16 and with GPUs RTX 3060 and 3090. 
