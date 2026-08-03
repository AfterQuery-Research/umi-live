import json, sys, time
import rclpy
from rclpy.node import Node

rclpy.init()
node = Node("pose_tap")

target = None
deadline = time.time() + 120
while target is None and time.time() < deadline:
    cands = [n for n, t in node.get_topic_names_and_types() if "nav_msgs/msg/Odometry" in t]
    if cands:
        cands.sort(key=lambda n: ("corrected" not in n, "scanend" in n, n))
        target = cands[0]
    if target is None:
        rclpy.spin_once(node, timeout_sec=0.5)
if target is None:
    print(json.dumps({"error": "no odometry topic found"}), flush=True)
    sys.exit(1)
print(json.dumps({"info": f"subscribed {target}"}), flush=True)

from nav_msgs.msg import Odometry

def cb(m):
    p = m.pose.pose.position
    q = m.pose.pose.orientation
    print(json.dumps({
        "t": m.header.stamp.sec + m.header.stamp.nanosec / 1e9,
        "x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4),
        "qx": round(q.x, 4), "qy": round(q.y, 4), "qz": round(q.z, 4), "qw": round(q.w, 4),
    }), flush=True)

node.create_subscription(Odometry, target, cb, 10)
rclpy.spin(node)
