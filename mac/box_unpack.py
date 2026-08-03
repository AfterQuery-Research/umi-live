import socket, struct, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
r = sys.stdin.buffer
n = 0
while True:
    hdr = r.read(4)
    if len(hdr) < 4:
        break
    port, ln = struct.unpack("!HH", hdr)
    payload = r.read(ln)
    if len(payload) < ln:
        break
    s.sendto(payload, ("127.0.0.1", port))
    n += 1
    if n % 5000 == 0:
        print(f"relayed {n} pkts", file=sys.stderr, flush=True)
