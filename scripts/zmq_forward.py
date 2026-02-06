"""This forwards the Orbbec Camera from an ROS topic via a ZMQ socket.

This is done because ZMQ imposes significantly less dependency restrictions that ROS.
"""
import os
import socket
import struct
import argparse

import zmq

import rospy
import ros_numpy
from sensor_msgs.msg import Image, CameraInfo



def bold_text(txt):
    return "\033[1m\033[34m{}\033[0m".format(txt)

def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip



parser = argparse.ArgumentParser("Set up a ROS-to-ZMQ forwarding node")
parser.add_argument("-p", "--port", type=int, help="Port to forward to", default=5000)
parser.add_argument("-n", "--node-name", type=str, help="Name of node within ROS", default="zmq_forward")
parser.add_argument("-u", "--ros-master-uri", type=str, help="ROS_MASTER_URI - if relevant")
parser.add_argument("-c", "--count", action="store_true", help="Prints seq for color image")
args = parser.parse_args()



context = zmq.Context()
zmq_socket  = context.socket(zmq.PUB)
zmq_socket.bind("tcp://*:{}".format(args.port))
zmq_socket.setsockopt(zmq.SNDHWM, 1)



def forward_img(topic, msg):
    frame_id = msg.header.frame_id.encode()

    meta = struct.pack("<IIIIIiI",
        msg.header.seq,
        msg.header.stamp.secs,
        msg.header.stamp.nsecs,
        msg.height,
        msg.width,
        msg.step,
        len(frame_id))

    zmq_socket.send(topic + meta + frame_id + msg.data)

    if args.count and topic == b"C":
        print(msg.header.seq)

def forward_info(topic, msg):
    frame_id = msg.header.frame_id.encode()
    
    meta = struct.pack("<III9dI",
        msg.header.seq,
        msg.header.stamp.secs,
        msg.header.stamp.nsecs,
        *msg.K,
        len(frame_id))
    
    zmq_socket.send(topic + meta + frame_id)



def color_callback(msg):
    forward_img(b"C", msg)

def depth_callback(msg):
    forward_img(b"D", msg)

def info_callback(msg):
    forward_info(b"I", msg)



if args.ros_master_uri:
    os.environ["ROS_MASTER_URI"] = args.ros_master_uri

rospy.init_node(args.node_name)

color_sub = rospy.Subscriber("/camera/color/image_raw", Image, color_callback, queue_size = 1)
depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback, queue_size = 1)
info_sub  = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, info_callback, queue_size = 1)

print("Spinning up ZMQ forwarding. {port:", bold_text(str(args.port)), 
                                   ", ip:", bold_text(str(get_lan_ip())), "}")

rospy.spin()