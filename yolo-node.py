# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

import signal, argparse, sys, threading, math

import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

from cv_bridge import CvBridge

import rospy, ros_numpy
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

# Constants
# 
# Note: These are ripped from the ultralytics project.
#       See: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/plotting.py
COL_TORSO = ColorRGBA(r=123./255., b=104./244., g=0.,        a=1.0) # "16"
COL_ARMS  = ColorRGBA(r=  1.,      b=128./255., g=0.,        a=1.0) # "0"
COL_HEAD  = ColorRGBA(r=204./255., b=0.,        g=237./255., a=1.0) # "7"
COL_LEGS  = ColorRGBA(r=51./255.,  b=1.,        g=154./255., a=1.0) # "9"

JOINTS = [[COL_HEAD, "Nose"],    # 0 - Corresponds to "1" from Ultralytics implementations
          [COL_HEAD, "Left Eye"],
          [COL_HEAD, "Right Eye"],
          [COL_HEAD, "Left Ear"],
          [COL_HEAD, "Right Ear"],
          [COL_ARMS, "Left Shoulder"], # 5
          [COL_ARMS, "Right Shoulder"], 
          [COL_ARMS, "Left Elbow"],
          [COL_ARMS, "Right Elbow"], 
          [COL_ARMS, "Left Wrist"],
          [COL_ARMS, "Right Wrist"],
          [COL_TORSO, "Left Hip"], # 11
          [COL_TORSO, "Right Hip"],
          [COL_LEGS,  "Left Knee"], # 13
          [COL_LEGS,  "Right Knee"],
          [COL_LEGS,  "Left Ankle"],
          [COL_LEGS,  "Right Ankle"]]

LIMBS = [[0, 1,   COL_HEAD],  # Nose -> Eyes
         [0, 2,   COL_HEAD],
         [1, 3,   COL_HEAD],  # Eyes -> Ears
         [2, 4,   COL_HEAD],
         [3, 5,   COL_HEAD],  # Ears -> Shoulders
         [4, 6,   COL_HEAD],
         [5, 6,   COL_ARMS],  # Shoulders
         [5, 7,   COL_ARMS],  # Shoulders -> Elbows
         [6, 8,   COL_ARMS], 
         [7, 9,   COL_ARMS],  # Elbows -> Wrists
         [8, 10,  COL_ARMS],
         [5, 11,  COL_TORSO], # Shoulders -> Hips
         [6, 12,  COL_TORSO],
         [11, 12, COL_TORSO], # Hips
         [11, 13, COL_LEGS],  # Hips -> Knees
         [12, 14, COL_LEGS], 
         [13, 15, COL_LEGS],  # Knees -> Ankles
         [14, 16, COL_LEGS]]

# Visualization helpers
def get_limb_marker(kpts, marker_id):
    """Takes a list of 3D coordinates for each keypoint (use a placeholder if N/A), and whether the keypoint is visible. Note: Header should be set on the returned marker."""
    marker = Marker()
    marker.type   = Marker.LINE_LIST
    marker.action = Marker.ADD
    for (idx1, idx2, col) in LIMBS:
        if kpts[idx1] is not None and kpts[idx2] is not None:
            marker.points.append(Point(*kpts[idx1]))
            marker.points.append(Point(*kpts[idx2]))
            marker.colors.append(col)
            marker.colors.append(col)
    marker.id = marker_id
    marker.scale.x = 0.005
    marker.scale.y = 0.001
    marker.scale.z = 0.001
    marker.pose.orientation.x = 0.
    marker.pose.orientation.y = 0.
    marker.pose.orientation.z = 0.
    marker.pose.orientation.w = 1.
    return marker 

def get_point_marker(kpts, marker_id):
    """Same as limb marker, but for keypoints themselves."""
    marker = Marker()
    marker.type   = Marker.POINTS
    marker.action = Marker.ADD
    for i, (col, _) in enumerate(JOINTS):
        if kpts[i] is not None:
            marker.points.append(Point(*kpts[i]))
            marker.colors.append(col)
    marker.id = marker_id
    marker.scale.x = 0.005
    marker.scale.y = 0.001
    marker.scale.z = 0.001
    marker.pose.orientation.x = 0.
    marker.pose.orientation.y = 0.
    marker.pose.orientation.z = 0.
    marker.pose.orientation.w = 1.
    return marker

def get_slice(dimg, bbox, shape=None):
    if shape:
        if isinstance(shape, int):
            shape = (shape, shape)
        x_min, x_max = bbox[0] - shape[0], bbox[0] + shape[0]
        x_min, x_max = bbox[0] - shape[0], bbox[0] + shape[0]
        y_min, y_max = bbox[1] - shape[1], bbox[1] + shape[1]
    else:
        x_min, y_min, x_max, y_max = bbox

    if x_min < 0:             x_min = 0
    if y_min < 0:             y_min = 0
    if x_max > dimg.shape[0]: x_max = dimg.shape[1] + 1
    if y_max > dimg.shape[1]: y_max = dimg.shape[1] + 1
    # print(f"Taking slice from {x_min}, {y_min} to {x_max}, {y_max}")
    return dimg[y_min: y_max, x_min: x_max]

def project_point(dimg, u, v):
    roi = get_slice(dimg, (u, v), 25)
    return np.mean(roi[np.where(roi != 0.)])

# Pinhole camera model
def get_pinhole_xy(cam_info, u, v, z, /):
    K  = cam_info.K
    fx = K[0+0] # 0,0
    fy = K[3+1] # 1,1
    cx = K[0+2] # 0,2
    cy = K[3+2] # 1,2
    x  = (u - cx) * z / fx
    y  = (v - cy) * z / fy
    return x, y

# NOTE - PoseEst2D assumes for every image_raw received, a corresponding camera_info is received
class PoseEst2D:
    def __init__(self, bridge, *, sub="/camera", pub="/yolo", model="yolov8m-pose.pt", preview=False, simple=False, cpu_only=False):
        if not "-pose.pt" in model:
            raise ValueError("Model must be pre-trained and of type '-pose'!")
        self.lock    = threading.Lock()
        self.bridge  = bridge
        self.pose_model   = YOLO(model)
        if cpu_only:
            self.pose_model = self.pose_model.to('cpu')
        self.preview = preview
        self.simple  = simple
        self.img_sub = rospy.Subscriber(f"{sub}/color/image_raw",
                                        Image,
                                        self.img_cb,
                                        queue_size = 1)
        self.info_sub = rospy.Subscriber(f"{sub}/color/camera_info",
                                         CameraInfo,
                                         self.info_cb,
                                         queue_size = 1) 
        self.depth_sub = rospy.Subscriber(f"{sub}/depth/image_raw",
                                          Image,
                                          self.depth_cb,
                                          queue_size = 1)
        self.img_pub  = rospy.Publisher(f"{pub}/yolo2D/image_raw",
                                        Image,
                                        queue_size = 1)
        self.info_pub = rospy.Publisher(f"{pub}/yolo2D/camera_info",
                                        CameraInfo,
                                        queue_size = 1)
        self.kpts_pub = rospy.Publisher(f"{pub}/kpts",
                                        MarkerArray,
                                        queue_size = 1)
        self.depth       = None
        self.results     = None
        self.frame_id    = None
        self.camera_info = None

    def depth_cb(self, data):
        print("Depth")
        self.depth = ros_numpy.numpify(data)

    def img_cb(self, data):
        self.frame_id = data.header.frame_id
        self.results  = self.pose_model.predict(ros_numpy.numpify(data), verbose=False)
        if self.preview:
            cv2.imshow("YOLO-pose", cv2.cvtColor(self.results[0].plot(), cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass # TODO - SIGINT equivalent...
        self.pub()

    def info_cb(self, data):
        # TODO - Sometimes this is an object with args (None,) - test and you'll quickly see the TypeError that occurs
        self.camera_info = data
        self.pub()

    def pub(self, threshold = 0.25):
        with self.lock:
            if self.results is not None \
            and self.frame_id is not None \
            and self.camera_info is not None \
            and self.depth is not None:
                # 1) HPE using YOLO - publish result
                img = self.bridge.cv2_to_imgmsg(self.results[0].plot())
                img.header.frame_id = self.frame_id
                self.img_pub.publish(img)
                if self.camera_info.header is not None:
                    self.info_pub.publish(self.camera_info)
                if not self.simple:
                    # 2) Project points onto depth map
                    markers = MarkerArray()
                    for result in self.results:
                        if result.keypoints.data is None:
                            continue
                        kpts     = []
                        for person in range(result.keypoints.data.shape[0]):
                           # Step 1 if hips and shoulders present, use them to calculate "depth"
                           depths_main = []
                           depths_sec  = []
                           for row in range(result.keypoints.data.shape[1]):
                               u_raw, v_raw, conf = result.keypoints.data[person, row]
                            #    print(conf)
                               if conf.item() > threshold:
                                    u, v = int(u_raw), int(v_raw)
                                    # d = self.depth[v, u] / 1000.0
                                    # print(f"Row {row}")
                                    d = project_point(self.depth, u, v) / 1000.
                                    if row in [5, 6, 11, 12]:
                                        depths_main.append(d)
                                    else:
                                        depths_sec.append(d)
                                    kpts.append([(u - self.results[0].orig_shape[1] // 2) / 1000.,
                                                 (v - self.results[0].orig_shape[0] // 2) / 1000.,
                                                 0.])
                                    # kpts.append([*get_pinhole_xy(self.camera_info, u/1000., v/1000., 0), d])
                               else:
                                    kpts.append(None)
                           if len(kpts) > 0:
                                d = 0.
                                if depths_main:
                                    d = np.mean(depths_main)
                                elif depths_sec:
                                    d = np.mean(depths_sec)
                                for row in range(len(kpts)):
                                    if kpts[row] is not None:
                                        kpts[row][2] = d
                        kpt_markers  = get_point_marker(kpts, 2*person)
                        limb_markers = get_limb_marker(kpts, 2*person+1)
                        kpt_markers.header.frame_id = self.frame_id
                        limb_markers.header.frame_id = self.frame_id
                        markers.markers.append(kpt_markers)
                        markers.markers.append(limb_markers)
                    self.kpts_pub.publish(markers)

                self.results, self.frame_id, self.camera_info, self.depth = None, None, None, None

# Register SIGINT kill
def kill_loop(sig, frame):
    rospy.signal_shutdown("User requested shutdown")
    sys.exit(0)

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create an ROS node to publish human pose estimation results using YOLO models.")
    parser.add_argument("-t", "--topic",
                        type=str,
                        default="/camera",
                        help="The topic root for the RGB-D camera.")
    parser.add_argument("-m", "--model",
                        type=str,
                        default="yolov8m",
                        help="The version/generation of YOLO models to use.")
    parser.add_argument("-p", "--preview",
                        action="store_true",
                        help="Set this flag to show the results in separate windows.")
    parser.add_argument("-s", "--simple",
                        action="store_true",
                        help="Set this flag to only produce the 2D preview.")    
    parser.add_argument("-c", "--cpu",
                        action="store_true",
                        help="Set this flag to only load model using CPU.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, kill_loop)

    rospy.init_node("yolo")
    bridge = CvBridge()

    hpe = PoseEst2D(bridge, preview=args.preview, sub=f"{args.topic}", model=f"{args.model}-pose.pt", simple=args.simple, cpu_only=args.cpu)

    rospy.spin()
