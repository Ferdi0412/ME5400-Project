# ME5400-Project
The goal of this project is to set up a person tracking and Human Pose Estimation system in ROS 1, for use in an autonomous person-following robot. This is part of my ME5400 project during my MSc in Robotics at NUS.

## Person Tracking
This should provide the person's location in "2D", in terms of distance and direction from image/robot.

* Relevant messages:
    1) `PointStamped` 
    2) `PoseStamped`
    3) `Marker` using `CYLINDER` and `0 < alpha < 1`

## Human Pose Estimation
Human Pose Estimation (**HPE**) provides **keypoints** representing a person's joints. There are various representations for this, one of the most commonly used being the **COCO-pose** format, where there are **17 keypoints**, representing the nose, and either eye, ear, shoulder, elbow, wrist, hip joint, knee, and ankle.

* Relevant messages:
    1) `Image` for easy visuzlization - must also have `CameraInfo` to display in *RVIZ*, and also have the `Image.header.frame_id` set
    2) `Marker` or `MarkerArray` using `POINTS`
    3) "Proper implementation" TBD

* Initial approach: Use the YOLO-pose models to get a 2D HPE, then project those points onto the **Orbbec's** depth image in the same frame.
* Later approach 1: Test some 3D HPE models, to get keypoints in 3D relative to each other, then project the position onto the depth map, retaining the 3D relations from the model.
* Later approach 2: Test Kinect-like models, which do not rely on RGB, only on depth, to estimate position. There are several libraries which should implement this, including **Orbbec-SDK-K4A-Wrapper** and Orbbec's **OpenNI_SDK**. Also check out Orbbec's **Azure-Kinect-Samples**
* Alternative approach to explore: Use 2 2D or 3D HPE models, one on either of the stereo IR cameras, then triangluate their position in 3D

# Setup
Assumes **Ubuntu 20.04** (Fossy Focal)  installation.

## ROS 1 (Noetic)
The following summarizes the [installation instructions](https://wiki.ros.org/noetic/installation/Ubuntu):
1) Initial setup
```bash
sudo sh -c 'echo "deb https://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```
2) Installation
```bash
sudo apt update
sudo apt install ros-noetic-desktop
```
3) Source ROS - either source in **every terminal** using ROS operations, or add to **.bashrc**
```bash
# Option 1: Source for every new terminal
source /opt/ros/noetic/setup.bash

# Option 2: Add to .bashrc to auto-source for every subsequent terminal
# NOTE: Careful to use '>>' and NOT '>' as the latter might break some terminal functions
echo "" >> ~/.bashrc
echo "# Auto-source ROS" >> ~/.bashrc
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Test that it works
roscore
```

## Python
I am using **miniconda** to control the environment.

1) Install `conda` following [instructions](https://anaconda.com/docs/getting-started/miniconda/install#macros-linux-installation) (Note: This is for the NVIDIA Jetson which uses an Aarch64 processor)
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Minconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

2) Source miniconda - 2 options as with ROS
```bash
# Option 1 - Source for every new terminal
source ~/miniconda3/bin/activate

# Option 2 - Append to .bashrc
echo "" >> ~/.bashrc
echo "# Auto-source conda" >> ~/.bashrc
echo "source ~/miniconda3/bin/activate" >> ~/.bashrc
```

3) Install core libraries
```bash
# Note - I have forgotten if this first line is run before or after creating and activating the environment
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages

# Set up environment
conda create -n me5400-project python=3.8
conda activate me5400-project
conda install pip

# Install required libraries
pip install -r requirements.txt

# Install ros_numpy
sudo apt install ros-noetic-numpy
```

4) Install Human Pose Estimation libraries - **Note:** the `--constraint constraints.txt` argument is important to avoid `numpy` version conflicts between the `ultralytics` library and `ros_numpy`. 
```bash
conda activate me5400-project
pip install -r req-yolo.txt --constraint constraints.txt
```

* YOLO compatability:
    - ROS Noetic uses `numpy<=1.20`, however only YOLO models up to YOLO**v8**.


## Orbbec ROS 1 SDK
This was developed using an **Orbbec Gemini 335**, so the **Orbbec SDK for ROS 1** was used. The following are the summarized [setup instructions](https://github.com/orbbec/OrbbecSDK_ROS1), and assume the appropriate **ROS** setup:

1) Clone and build repository
```bash
mkdir -p ~/orbbec_ws/src
git clone https://github.com/orbbec/OrbbecSDK_ROS1.git ~/orbbec_ws/src/OrbbecSDK_ROS1
cd ~/orbbec_ws
catkin_make
source ./devel/setup.bash
roscd orbbec_camera
sudo bash ./scripts/install_udev_rules.sh
```

2) Source Orbbec ROS - in **every terminal** used to launch Orbbec camera nodes
```bash
cd ~/orbbec_ws
source ./devel/setup.bash
```

3) Start camera
```bash
roslaunch orbbec_camera gemini_330_series.launch
```

4) Open RVIZ
```bash
rviz -f camera_depth_frame

# To get alternatives to "camera_depth_frame"
rosrun tf view_frames # Open this to see options
```


