#!/bin/bash

# ===============
# === STYLING ===
# ===============
# Print as `echo -e "$PRE ...`
DIM="\033[2m"
BLU="\033[1m\033[34m"
YEL="\033[1m\033[33m"
RED="\033[1m\033[31m"
END="\033[0m"
PRE="${DIM}>> ${END}"

# ============
# === MAIN ===
# ============
MY_IP=$(hostname -I | awk '{print $1}')

export ROS_IP=$MY_IP
export ROS_MASTER_URI=http://$MY_IP:11311

# Setup ros-source.sh
echo -e "export ROS_MASTER_URI=http://${MY_IP}:11311" > orbbec-source.sh
echo -e "export ROS_IP=ROS_IP=\$(hostname -I | awk '{print \$1}')" >> orbbec-source.sh

# Setup start-orbbec.sh
echo -e "${PRE}To use the correct ROS CORE, run the following:"
echo -e "${PRE}  ${BLU}export${END} ROS_MASTER_URI=http://${BLU}${MY_IP}${END}:11311"
echo -e "${PRE}  ${BLU}export${END} ROS_IP=\$(hostname -I | awk '{print \$1}')"
echo -e "${PRE}Alternatively, copy and source ${DIM}orbbec-source.sh"
echo -e "${PRE}  ${BLU}source${END} ros-source.sh"
echo

cd ~/orbbec_ws
source ./devel/setup.bash
roslaunch orbbec_camera gemini_330_series.launch
