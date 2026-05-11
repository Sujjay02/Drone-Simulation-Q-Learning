#!/usr/bin/env python3
"""
capture_frames.py
-----------------
Subscribes to F450 RGBD camera topics and saves frames as PNGs for offline YOLO testing.

Runs inside the MRS Apptainer container after the 3-drone simulation is up and the
drones are hovering. Saves frames to ~/yolo_test_frames/ on the host (the home dir is
mounted into the container, so the files persist).

Usage (inside container):
    # Discover what camera topics are actually being published
    python3 capture_frames.py --discover

    # Capture from uav1's color camera (default topic guess)
    python3 capture_frames.py --uav uav1 --num 20 --interval 1.0

    # Override the topic if discover shows a different name
    python3 capture_frames.py --topic /uav1/rgbd/color/image_raw --num 20

    # Capture from all 3 drones
    python3 capture_frames.py --all --num 10
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


DEFAULT_OUT_DIR = os.path.expanduser("~/yolo_test_frames")


class FrameCapture:
    def __init__(self, topic, out_dir, num_frames, interval, label):
        self.topic = topic
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.interval = interval
        self.label = label
        self.bridge = CvBridge()
        self.captured = 0
        self.last_capture = 0.0
        self.latest_msg = None

        rospy.Subscriber(topic, Image, self._image_cb, queue_size=1)
        rospy.loginfo(f"[{label}] Subscribed to {topic}")
        rospy.loginfo(f"[{label}] Saving up to {num_frames} frames to {out_dir} "
                      f"(every {interval}s)")

    def _image_cb(self, msg):
        self.latest_msg = msg

    def run(self):
        rate = rospy.Rate(10)
        wait_start = time.time()
        while not rospy.is_shutdown() and self.captured < self.num_frames:
            if self.latest_msg is None:
                if time.time() - wait_start > 10:
                    rospy.logwarn(
                        f"[{self.label}] No images on {self.topic} after 10s. "
                        f"Confirm topic with: rostopic list | grep image"
                    )
                    wait_start = time.time()
                rate.sleep()
                continue

            now = time.time()
            if now - self.last_capture < self.interval:
                rate.sleep()
                continue

            try:
                cv_img = self.bridge.imgmsg_to_cv2(
                    self.latest_msg, desired_encoding="bgr8"
                )
            except CvBridgeError as e:
                rospy.logerr(f"[{self.label}] cv_bridge error: {e}")
                rate.sleep()
                continue

            timestamp = int(now * 1000)
            fname = self.out_dir / f"{self.label}_{timestamp}.png"
            cv2.imwrite(str(fname), cv_img)
            self.captured += 1
            self.last_capture = now
            rospy.loginfo(
                f"[{self.label}] Saved {self.captured}/{self.num_frames}: "
                f"{fname.name} ({cv_img.shape[1]}x{cv_img.shape[0]})"
            )
            rate.sleep()

        rospy.loginfo(f"[{self.label}] Done. {self.captured} frames in {self.out_dir}")


def discover_image_topics():
    try:
        result = subprocess.run(
            ["rostopic", "list"], capture_output=True, text=True, timeout=5
        )
        topics = [
            t for t in result.stdout.split("\n")
            if any(k in t.lower() for k in ("image", "rgbd", "camera"))
        ]
        print("Image-related topics currently published:")
        for t in sorted(topics):
            print(f"  {t}")
        if not topics:
            print("  (none found — is the simulation running with the modified F450?)")
    except Exception as e:
        print(f"Could not list topics: {e}")
        print("Try manually: rostopic list | grep -E 'image|rgbd|camera'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav", default="uav1", help="UAV name (uav1/uav2/uav3)")
    parser.add_argument(
        "--topic", default=None,
        help="Override image topic (default: /<uav>/rgbd/color/image_raw)",
    )
    parser.add_argument("--num", type=int, default=20, help="Frames to capture")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between captures")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--all", action="store_true",
                        help="Capture from all 3 drones in parallel")
    parser.add_argument("--discover", action="store_true",
                        help="List image topics and exit")
    args = parser.parse_args()

    if args.discover:
        discover_image_topics()
        return

    rospy.init_node("frame_capture", anonymous=True)

    if args.all:
        # Subscribe to all three but capture sequentially via a single shared rate
        captures = []
        for u in ("uav1", "uav2", "uav3"):
            t = f"/{u}/rgbd/color/image_raw"
            captures.append(FrameCapture(t, args.out, args.num, args.interval, u))
        # Round-robin: keep looping until each has its quota
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and any(c.captured < c.num_frames for c in captures):
            rate.sleep()
        return

    topic = args.topic or f"/{args.uav}/rgbd/color/image_raw"
    capture = FrameCapture(topic, args.out, args.num, args.interval, args.uav)
    capture.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
