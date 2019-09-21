# BodyScan
Measure your body fat percentage with just a single picture!

# Installation
### Using virtual environment
This code has been tested on Ubuntu, PyTorch 1.2, Python 3.6 and Nvidia GTX 940MX. It is recommended to setup a python virtual environment 
and install the following packages.

1. Clone the repo
2. Install the below:
   ```
   apt-get install tk-dev python-tk
   ```

3. Activate the virutal Install the required python packages in a virtual environment

   ```
   (pytorch)$ pip3 install torch torchvision 
   (pytorch)$ pip3 install scikit-image opencv-python pandas h5py
   (pytorch)$ pip3 install cffi
   (pytorch)$ pip3 install cython
   (pytorch)$ pip3 install requests
   (pytorch)$ pip3 install future
   ```
3. Build the NMS extension

   ```
   cd lib/
   python3 setup3.py build_ext --inplace
   ```
### Using docker

To be updated soon!
# Usage
### Run a demo
1. `python3 measure_body.py`  
   This takes a sample picture from `data/inputs` and predicts the body fat percentage. 
### Estimate your own body fat percentage! 
1. **Instructions for taking pictures**  
The model will estimate your neck and waist circumference to predict your body fat percentage. So your neck and
waist area needs to be clearly visible in the picture. Also, the model works best when you are standing atleast 1 m 
away from the camera. Some examples:  

   **Good example**  
   
   ![Image](./data/inputs/204.jpg)

   **Bad examples**
   
2. Paste your picture in `data/inputs/`
3. Run `python3 measure_body.py --image_name <name_of_your_image>.jpg`  
   Your results are shown in the screen.

# Working
To be updated soon! 

# Acknowledgements
* Depth estimation code has been borrowed & modified from this [repo](https://github.com/google/mannequinchallenge)
  (implementation of this awesome [google AI](https://mannequin-depth.github.io/) paper). 
* Retinanet code has been borrowed & modified from [this PyTorch](https://github.com/yhenon/pytorch-retinanet) 
  implementation.
* NMS code from [here](https://github.com/huaifeng1993/NMS).
