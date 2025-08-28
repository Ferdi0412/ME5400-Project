import signal, argparse, sys, threading

import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO

from cv_bridge import CvBridge

import rospy, ros_numpy
from sensor_msgs.msg import Image, CameraInfo

# NOTE - PoseEst2D assumes for every image_raw received, a corresponding camera_info is received
class PoseEst2D:
    def __init__(self, bridge, *, sub="/camera/color", pub="/yolo", model="yolov8m-pose.pt", preview=False):
        if not "-pose.pt" in model:
            raise ValueError("Model must be pre-trained and of type '-pose'!")
        self.lock    = threading.Lock()
        self.bridge  = bridge
        self.pose_model   = YOLO(model)
        self.preview = preview
        self.img_sub = rospy.Subscriber(f"{sub}/image_raw",
                                        Image,
                                        self.img_cb,
                                        queue_size = 1)
        self.info_sub = rospy.Subscriber(f"{sub}/camera_info",
                                         CameraInfo,
                                         self.info_cb,
                                         queue_size = 1) 
        self.img_pub  = rospy.Publisher(f"{pub}/yolo2D/image_raw",
                                        Image,
                                        queue_size = 1)
        self.info_pub = rospy.Publisher(f"{pub}/yolo2D/camera_info",
                                        CameraInfo,
                                        queue_size = 1)
        self.results     = None
        self.frame_id    = None
        self.camera_info = None

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

    def pub(self):
        with self.lock:
            if self.results is not None and self.frame_id is not None and self.camera_info is not None:
                img = self.bridge.cv2_to_imgmsg(self.results[0].plot())
                img.header.frame_id = self.frame_id
                self.img_pub.publish(img)
                if self.camera_info.header is not None:
                    self.info_pub.publish(self.camera_info)
                self.results, self.frame_id, self.camera_info = None, None, None

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
    args = parser.parse_args()

    signal.signal(signal.SIGINT, kill_loop)

    rospy.init_node("yolo")
    bridge = CvBridge()

    hpe = PoseEst2D(bridge, preview=args.preview, sub=f"{args.topic}/color", model=f"{args.model}-pose.pt")

    rospy.spin()
