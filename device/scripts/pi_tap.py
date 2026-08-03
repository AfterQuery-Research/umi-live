"""Raw lidar UDP tap for the LIVE-STREAMING path (the Pi stays ROS-free).

Binds the lidar's UDP ports, and muxes every datagram to stdout as
``struct.pack("!HH", port, len) + payload``. The Mac drives this over SSH and
pipes it into ``mac/box_unpack.py`` inside the GLIM container, which demuxes it
back to loopback UDP for rslidar_sdk -> GLIM. No decode happens here; the Pi only
forwards bytes.

    python3 pi_tap.py [port ...]        # default: 6699 7788 6688 (RoboSense Airy)
"""
import select
import socket
import struct
import sys

PORTS = [int(p) for p in sys.argv[1:]] or [6699, 7788, 6688]
socks = []
for port in PORTS:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
    s.bind(("0.0.0.0", port))
    s.setblocking(False)
    socks.append(s)

out = sys.stdout.buffer
sent = 0
while True:
    readable, _, _ = select.select(socks, [], [], 1.0)
    for s in readable:
        try:
            while True:
                data, _ = s.recvfrom(2048)
                port = s.getsockname()[1]
                out.write(struct.pack("!HH", port, len(data)))
                out.write(data)
                sent += 1
                if sent % 2000 == 0:
                    print(f"sent {sent}", file=sys.stderr, flush=True)
        except BlockingIOError:
            pass
    out.flush()
