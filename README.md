## Manual_tf_GPU

# This is a step by step guide to install latest version of TensorFlow on GPU

*07/01/2022*

*Tested successfully on Windows 10 & NVIDIA Quadro T1000 with Max-Q Design*

## Why manual installation ?
After many tries to get things done using "conda install tensorflow-gpu", It failed and gives a ton of errors. So I propose doing things manually to save time. And to make sure everything is well installed. This is a manual installation guide of TensorFlow to avoid any kind of errors.

## Step 1: Hardware check
* Using DxDialog check GPU series and version
 
## Step 2: Install driver (I already have it)
*	https://www.nvidia.com/download/index.aspx?lang=en-us

## Step 3: Install Visual Studio IDE
*	https://visualstudio.microsoft.com/fr/vs/

## Step 4: Install CUDA Toolkit
*	Version 11.5
*	https://developer.nvidia.com/cuda-downloads

## Step 5: Install cuDNN
*	Create account or log in
*	Make sure to choose the version suitable with CUDA 11.5
*	V 8.3.1
*	https://developer.nvidia.com/cudnn
*	Download the zip folder. Decompress it in C:\tools\

## Step 6: Add environment variables
*	https://www.tensorflow.org/install/gpu
```
1.	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin
2.	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\libnvvp
3.	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\include
4.	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64
5.	C:\tools\cuda-windows-x86_64-8.3.1.22_cuda11.5-archive\bin
6.	C:\tools\cuda-windows-x86_64-8.3.1.22_cuda11.5-archive\include
```
 
(*) check paths depending on your OS and used versions !

## Step 7: Install Miniconda3
*	https://docs.conda.io/en/latest/miniconda.html
*	Latest Miniconda Installer Links => Miniconda3 Windows 64-bit
*	Conda version 4.11.0
*	Python 3.9.5
*	Add Miniconda3 to my path environment variable


## Step 8: Install Jupyter
*	Go to Anaconda Prompt (base)
```
# conda install -y jupyter
```
*	Test by launching a notebook
```
# jupyter notebook
```

## Step 9: Create a new environment
*	Create a new environment called: tensorflow
```
# conda create -y --name tensorflow python=3.9
```
* Activate the new environment
```
# conda activate tensorflow
```

## Step 10: Jupyter Kernel
*	Use Anaconda Prompt
*	Navigate to ‘tensorflow’ environment
```
# conda install ipykernel
# python -m ipykernel install --user --name tensorflow --display-name "Python 3.9 (tensorflow)"
```
*	Check the environment in Jupyter notebook

## Step 11: Install TensorFlow
```
# pip install tensorflow
```
*	Installed latest version at this day: 2.7
*	Check installation
```
> import tensorflow as tf
> print(tf.__version__)
> print(len(tf.config.list_physical_devices('GPU'))>0) # should return True
```

## Step 12: Install additional packages for our project
```
(tensorflow) # pip install pandas
(tensorflow) # pip install pillow
(tensorflow) # pip install tensorflow-addons
(tensorflow) # pip install pydot
(tensorflow) # pip install matplotlib
```

## References
*	https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup2.ipynb
*	https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup.ipynb
*	https://www.youtube.com/watch?v=OEFKlRSd8Ic

